using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>granitemoehybrid</c> (Granite-4.0-H) → canonical <c>hybrid</c> (documented on <c>HybridAdapter</c>). Layers are Mamba-2 or attention
/// (<c>layer_types</c>); every layer has the shared MLP (fused <c>input_linear</c> = gate rows then up rows) and, when <c>num_local_experts</c> &gt; 0, routed experts.
/// </summary>
internal sealed partial class GraniteHybridHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "hybrid";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["granitemoehybrid"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        int hidden = c.Int("hidden_size");
        int heads = c.Int("num_attention_heads");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = c.Int("num_hidden_layers"),
            [a + "hidden_size"] = hidden,
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-5),
            [a + "embedding_scale"] = (float)(c.GetDouble("embedding_multiplier") ?? 1.0),
            [a + "residual_scale"] = (float)(c.GetDouble("residual_multiplier") ?? 1.0),
            [a + "logit_scale"] = 1f / (float)(c.GetDouble("logits_scaling") ?? 1.0),
            [a + "attention.head_count"] = heads,
            [a + "attention.kv_head_count"] = c.GetInt("num_key_value_heads") ?? heads,
            [a + "attention.head_dim"] = hidden / heads,
            [a + "attention.scale"] = (float)(c.GetDouble("attention_multiplier") ?? 1.0 / Math.Sqrt(hidden / heads)),
            [a + "attention.rope"] = c.GetString("position_embedding_type") == "rope",
            [a + "rope.base"] = (float)(c.RopeDouble("rope_theta", "rope_theta") ?? 10000.0),
            [a + "ssm.conv_kernel"] = c.Int("mamba_d_conv"),
            [a + "ssm.state_size"] = c.Int("mamba_d_state"),
            [a + "ssm.group_count"] = c.Int("mamba_n_groups"),
            [a + "ssm.head_count"] = c.Int("mamba_n_heads"),
            [a + "ssm.head_dim"] = c.Int("mamba_d_head"),
            [a + "mlp.activation"] = c.GetString("hidden_act") ?? "silu",
        };
        HfMappingHelpers.AddContextLength(md, a, c);
        if ((c.GetInt("num_local_experts") ?? 0) > 0)
        {
            md[a + "moe.expert_count"] = c.Int("num_local_experts");
            md[a + "moe.expert_used_count"] = c.Int("num_experts_per_tok");
        }

        return md;
    }

    [GeneratedRegex(@"^model\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    /// <summary>Per-layer tensors that are only renamed, HF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["input_layernorm.weight"] = "input_norm.weight",
        ["post_attention_layernorm.weight"] = "post_attention_norm.weight",
        ["self_attn.q_proj.weight"] = "attention.q.weight",
        ["self_attn.k_proj.weight"] = "attention.k.weight",
        ["self_attn.v_proj.weight"] = "attention.v.weight",
        ["self_attn.o_proj.weight"] = "attention.o.weight",
        ["block_sparse_moe.router.layer.weight"] = "mlp.router.weight",
        ["block_sparse_moe.output_linear.weight"] = "mlp.experts.down.weight",
    };

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var collector = new HfWeightCollector();
        var layersWithExperts = tensors.Keys.Where(n => n.Contains(".block_sparse_moe.router.", StringComparison.Ordinal))
            .Select(n => LayerName().Match(n).Groups[1].Value).ToHashSet();
        foreach (var (name, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            var w = tensor.ToWeightInfo();
            var m = LayerName().Match(name);
            bool mapped = m.Success
                ? TryMapLayerTensor(m.Groups[1].Value, m.Groups[2].Value, w, collector, layersWithExperts.Contains(m.Groups[1].Value))
                : TryMapGlobalTensor(c, name, w, collector);
            if (!mapped)
            {
                collector.AddUnmapped(name);
            }
        }

        collector.ThrowIfUnmapped("granitemoehybrid");
        return collector.Weights;
    }

    private static bool TryMapGlobalTensor(HfConfig c, string name, WeightInfo w, HfWeightCollector collector)
    {
        switch (name)
        {
            case "model.embed_tokens.weight":
                collector.Add(HfTransforms.Plain(w, "token_embedding.weight"));
                return true;
            case "model.norm.weight":
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

    private static bool TryMapLayerTensor(string layer, string suffix, WeightInfo w, HfWeightCollector collector, bool hasExperts)
    {
        string prefix = $"layers.{layer}.";
        if (s_layer.TryGetValue(suffix, out var renamed))
        {
            collector.Add(HfTransforms.Plain(w, prefix + renamed));
            return true;
        }

        if (suffix.StartsWith("mamba.", StringComparison.Ordinal))
        {
            return HybridHfTensors.TryMapSsm(prefix, suffix["mamba.".Length..], w, collector);
        }

        // The shared MLP is the dense MLP, or the shared expert next to routed experts.
        string mlp = hasExperts ? "mlp.shared." : "mlp.";
        switch (suffix)
        {
            case "shared_mlp.input_linear.weight":
                long half = w.Shape[0].Fixed / 2;
                collector.Add(HfTransforms.Plain(w.RowSlice("", 0, half), prefix + mlp + "gate.weight"));
                collector.Add(HfTransforms.Plain(w.RowSlice("", half, half), prefix + mlp + "up.weight"));
                return true;
            case "shared_mlp.output_linear.weight":
                collector.Add(HfTransforms.Plain(w, prefix + mlp + "down.weight"));
                return true;
            case "block_sparse_moe.input_linear.weight":
                SplitExpertGateUp(w, prefix, collector);
                return true;
            default:
                return false;
        }
    }

    /// <summary>Splits the fused [E, 2I, hidden] expert input projection into gate [E, I, hidden] (first half of the rows) and up (second half).</summary>
    private static void SplitExpertGateUp(WeightInfo w, string prefix, HfWeightCollector collector)
    {
        long experts = w.Shape[0].Fixed;
        long half = w.Shape[1].Fixed / 2;
        long columns = w.Shape[2].Fixed;
        var shape = new Shape(experts, half, columns);
        WeightInfo Half(string name, long firstRow) => WeightInfo.Derived(name, DataType.F32, shape, () =>
        {
            var values = HfTransforms.DecodeF32(w);
            var result = new float[experts * half * columns];
            for (long e = 0; e < experts; e++)
            {
                Array.Copy(values, (e * 2 * half + firstRow) * columns, result, e * half * columns, half * columns);
            }

            return HfTransforms.EncodeF32(result);
        });
        collector.Add(Half(prefix + "mlp.experts.gate.weight", 0));
        collector.Add(Half(prefix + "mlp.experts.up.weight", half));
    }
}
