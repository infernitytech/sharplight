using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>falcon_h1</c> → canonical <c>hybrid</c> (documented on <c>HybridAdapter</c>): attention and Mamba-2 in every layer, then a gated MLP.
/// The model's linear multipliers are folded into the weights as llama.cpp's converter does (SSM input multiplier and the per-section mup vector into
/// <c>ssm.in</c>, SSM output multiplier into <c>ssm.out</c>, attention input/key/output multipliers into q/k/v/o, MLP gate/down multipliers); the embedding and
/// lm-head multipliers become <c>embedding_scale</c> and <c>logit_scale</c>.
/// </summary>
internal sealed partial class FalconH1HfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "hybrid";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["falcon_h1"];

    private static double Multiplier(HfConfig c, string key) => c.GetDouble(key) ?? 1.0;

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        int heads = c.Int("num_attention_heads");
        int ssmHeads = c.Int("mamba_n_heads");
        int inner = c.GetInt("mamba_d_ssm") ?? c.Int("mamba_expand") * c.Int("hidden_size");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = c.Int("num_hidden_layers"),
            [a + "hidden_size"] = c.Int("hidden_size"),
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-5),
            [a + "embedding_scale"] = (float)Multiplier(c, "embedding_multiplier"),
            [a + "logit_scale"] = (float)Multiplier(c, "lm_head_multiplier"),
            [a + "attention.head_count"] = heads,
            [a + "attention.kv_head_count"] = c.GetInt("num_key_value_heads") ?? heads,
            [a + "attention.head_dim"] = c.GetInt("head_dim") ?? c.Int("hidden_size") / heads,
            [a + "attention.rope"] = true,
            [a + "rope.base"] = (float)(c.RopeDouble("rope_theta", "rope_theta") ?? 10000.0),
            [a + "ssm.conv_kernel"] = c.Int("mamba_d_conv"),
            [a + "ssm.state_size"] = c.Int("mamba_d_state"),
            [a + "ssm.group_count"] = c.Int("mamba_n_groups"),
            [a + "ssm.head_count"] = ssmHeads,
            [a + "ssm.head_dim"] = inner / ssmHeads,
            [a + "mlp.activation"] = c.GetString("hidden_act") ?? "silu",
        };
        if (c.GetBool("mamba_rms_norm") ?? false)
        {
            throw new NotSupportedException("falcon_h1: mamba_rms_norm=true (gated norm with weights) is not supported yet.");
        }

        HfMappingHelpers.AddContextLength(md, a, c);
        return md;
    }

    [GeneratedRegex(@"^model\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            var w = tensor.ToWeightInfo();
            var m = LayerName().Match(name);
            bool mapped = m.Success ? TryMapLayerTensor(c, $"layers.{m.Groups[1].Value}.", m.Groups[2].Value, w, collector) : TryMapGlobalTensor(c, name, w, collector);
            if (!mapped)
            {
                collector.AddUnmapped(name);
            }
        }

        collector.ThrowIfUnmapped("falcon_h1");
        return collector.Weights;
    }

    private static bool TryMapGlobalTensor(HfConfig c, string name, WeightInfo w, HfWeightCollector collector)
    {
        switch (name)
        {
            case "model.embed_tokens.weight":
                collector.Add(HfTransforms.Plain(w, "token_embedding.weight"));
                return true;
            case "model.final_layernorm.weight":
                collector.Add(HfTransforms.Plain(w, "output_norm.weight"));
                return true;
            case "lm_head.weight":
                if (!(c.GetBool("tie_word_embeddings") ?? false))
                {
                    collector.Add(HfTransforms.Plain(w, "lm_head.weight"));
                }

                return true;
            default:
                return false;
        }
    }

    private static bool TryMapLayerTensor(HfConfig c, string prefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        float attentionIn = (float)Multiplier(c, "attention_in_multiplier");
        float[] mlp = c.Get("mlp_multipliers") is { ValueKind: System.Text.Json.JsonValueKind.Array } m ? [.. m.EnumerateArray().Select(e => (float)e.GetDouble())] : [1f, 1f];
        switch (suffix)
        {
            case "input_layernorm.weight":
                collector.Add(HfTransforms.Plain(w, prefix + "input_norm.weight"));
                return true;
            case "pre_ff_layernorm.weight":
                collector.Add(HfTransforms.Plain(w, prefix + "post_attention_norm.weight"));
                return true;
            case "self_attn.q_proj.weight" or "self_attn.v_proj.weight":
                collector.Add(HybridHfTensors.Scaled(w, prefix + (suffix.Contains("q_proj") ? "attention.q.weight" : "attention.v.weight"), attentionIn));
                return true;
            case "self_attn.k_proj.weight":
                collector.Add(HybridHfTensors.Scaled(w, prefix + "attention.k.weight", attentionIn * (float)Multiplier(c, "key_multiplier")));
                return true;
            case "self_attn.o_proj.weight":
                collector.Add(HybridHfTensors.Scaled(w, prefix + "attention.o.weight", (float)Multiplier(c, "attention_out_multiplier")));
                return true;
            case "feed_forward.gate_proj.weight":
                collector.Add(HybridHfTensors.Scaled(w, prefix + "mlp.gate.weight", mlp[0]));
                return true;
            case "feed_forward.up_proj.weight":
                collector.Add(HfTransforms.Plain(w, prefix + "mlp.up.weight"));
                return true;
            case "feed_forward.down_proj.weight":
                collector.Add(HybridHfTensors.Scaled(w, prefix + "mlp.down.weight", mlp[1]));
                return true;
        }

        if (!suffix.StartsWith("mamba.", StringComparison.Ordinal))
        {
            return false;
        }

        return HybridHfTensors.TryMapSsm(prefix, suffix["mamba.".Length..], w, collector,
            (weight, name) => ScaledInputProjection(c, weight, name), (float)Multiplier(c, "ssm_out_multiplier"));
    }

    /// <summary>
    /// The input projection with the SSM input multiplier and the per-section mup multipliers (z, x, B, C, dt) folded in. Rows: z [inner] | x [inner] | B [G·N] | C [G·N] | dt [H].
    /// </summary>
    private static WeightInfo ScaledInputProjection(HfConfig c, WeightInfo w, string name)
    {
        int inner = c.GetInt("mamba_d_ssm") ?? c.Int("mamba_expand") * c.Int("hidden_size");
        int groupWidth = c.Int("mamba_n_groups") * c.Int("mamba_d_state");
        float input = (float)Multiplier(c, "ssm_in_multiplier");
        float[] section = c.Get("ssm_multipliers") is { ValueKind: System.Text.Json.JsonValueKind.Array } m
            ? [.. m.EnumerateArray().Select(e => (float)e.GetDouble())]
            : [1f, 1f, 1f, 1f, 1f];
        return HybridHfTensors.ScaledRows(w, name, row =>
        {
            int s = row < inner ? 0 : row < 2 * inner ? 1 : row < 2 * inner + groupWidth ? 2 : row < 2 * inner + 2 * groupWidth ? 3 : 4;
            return input * section[s];
        });
    }
}
