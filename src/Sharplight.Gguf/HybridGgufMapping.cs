using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>granitehybrid</c> (Granite-4.0-H), <c>falcon-h1</c>, <c>mamba2</c> and <c>lfm2</c> (gated short convolutions + attention) → canonical <c>hybrid</c> (documented on <c>HybridAdapter</c>).
/// A layer has attention when its GGUF key/value head count is positive; every layer of these models has the Mamba-2 mixer except the attention-only
/// layers of Granite-4.0-H (Falcon-H1 has both mixers in every layer, recognised by the SSM weights being present).
/// </summary>
internal sealed class HybridGgufMapping : IGgufArchitectureMapping
{
    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["granitehybrid", "falcon-h1", "mamba2", "lfm2"];
    /// <inheritdoc/>
    public string Architecture => "hybrid";

    /// <summary>The per-layer key/value head counts (a scalar is repeated for every layer).</summary>
    private static int[] KvHeadCounts(GgufMetadataView g, int layerCount)
        => g.Raw("attention.head_count_kv") is Array values && values.Length == layerCount
            ? Enumerable.Range(0, layerCount).Select(i => Convert.ToInt32(values.GetValue(i))).ToArray()
            : Enumerable.Repeat(g.Int("attention.head_count_kv", 0), layerCount).ToArray();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int layerCount = g.Int("block_count");
        int hidden = g.Int("embedding_length");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layerCount,
            [a + "hidden_size"] = hidden,
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
        };
        if (g.Has("ssm.inner_size"))
        {
            int ssmHeads = g.Int("ssm.time_step_rank");
            md[a + "ssm.conv_kernel"] = g.Int("ssm.conv_kernel");
            md[a + "ssm.state_size"] = g.Int("ssm.state_size");
            md[a + "ssm.group_count"] = g.Int("ssm.group_count");
            md[a + "ssm.head_count"] = ssmHeads;
            md[a + "ssm.head_dim"] = g.Int("ssm.inner_size") / ssmHeads;
        }

        if (g.Has("shortconv.l_cache"))
        {
            md[a + "shortconv.kernel"] = g.Int("shortconv.l_cache");
        }

        GgufMappingHelpers.AddContextLength(md, a, g);

        int kvHeads = KvHeadCounts(g, layerCount).Max();
        if (kvHeads > 0)
        {
            int headCount = g.Int("attention.head_count");
            md[a + "attention.head_count"] = headCount;
            md[a + "attention.kv_head_count"] = kvHeads;
            md[a + "attention.head_dim"] = g.Int("attention.key_length", hidden / headCount);
            md[a + "attention.scale"] = g.Float("attention.scale", 1f / MathF.Sqrt(g.Int("attention.key_length", hidden / headCount)));
            md[a + "rope.dim"] = g.Int("rope.dimension_count", g.Int("attention.key_length", hidden / headCount));
            md[a + "rope.base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase);
            // Granite-4.0-H attention is position-less (NoPE) unless the checkpoint is flagged as rope-finetuned; Falcon-H1 and LFM2 always rotate.
            md[a + "attention.rope"] = g.GgufArchitecture is "falcon-h1" or "lfm2" || g.Bool("rope.scaling.finetuned", false);
        }

        if (g.Has("embedding_scale"))
        {
            md[a + "embedding_scale"] = g.Float("embedding_scale");
        }

        if (g.Has("residual_scale"))
        {
            md[a + "residual_scale"] = g.Float("residual_scale");
        }

        if (g.Has("logit_scale"))
        {
            md[a + "logit_scale"] = 1f / g.Float("logit_scale");   // llama.cpp divides the logits by this value
        }

        if (g.Int("expert_count", 0) > 0)
        {
            md[a + "moe.expert_count"] = g.Int("expert_count");
            md[a + "moe.expert_used_count"] = g.Int("expert_used_count");
        }

        return md;
    }

    /// <summary>Per-layer tensors, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["ffn_norm"] = "post_attention_norm.weight",   // Falcon-H1 omits the suffix
        ["ssm_in.weight"] = "ssm.in.weight",
        ["ssm_out.weight"] = "ssm.out.weight",
        ["ssm_conv1d.weight"] = "ssm.conv1d.weight",
        ["ssm_conv1d.bias"] = "ssm.conv1d.bias",
        ["ssm_dt.bias"] = "ssm.dt_bias",
        ["ssm_a"] = "ssm.a",
        ["ssm_d"] = "ssm.d",
        ["ssm_norm.weight"] = "ssm.norm.weight",
        ["shortconv.in_proj.weight"] = "shortconv.in.weight",
        ["shortconv.conv.weight"] = "shortconv.conv.weight",
        ["shortconv.out_proj.weight"] = "shortconv.out.weight",
        ["attn_q_norm.weight"] = "attention.q_norm.weight",
        ["attn_k_norm.weight"] = "attention.k_norm.weight",
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_k.weight"] = "attention.k.weight",
        ["attn_v.weight"] = "attention.v.weight",
        ["attn_output.weight"] = "attention.o.weight",
        ["ffn_gate.weight"] = "mlp.gate.weight",
        ["ffn_up.weight"] = "mlp.up.weight",
        ["ffn_down.weight"] = "mlp.down.weight",
        ["ffn_gate_inp.weight"] = "mlp.router.weight",
        ["ffn_gate_exps.weight"] = "mlp.experts.gate.weight",
        ["ffn_up_exps.weight"] = "mlp.experts.up.weight",
        ["ffn_down_exps.weight"] = "mlp.experts.down.weight",
        ["ffn_gate_shexp.weight"] = "mlp.shared.gate.weight",
        ["ffn_up_shexp.weight"] = "mlp.shared.up.weight",
        ["ffn_down_shexp.weight"] = "mlp.shared.down.weight",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (name == "token_embd_norm.weight" && g.GgufArchitecture == "lfm2")
        {
            return "output_norm.weight";   // LFM2 calls its final norm the embedding norm
        }

        if (GgufTensorNames.MapGlobal(name) is { } globalName)
        {
            return globalName;
        }

        var m = GgufTensorNames.Block().Match(name);
        return m.Success && s_layer.TryGetValue(m.Groups[2].Value, out var c) ? $"layers.{m.Groups[1].Value}.{c}" : null;
    }

    /// <summary>A, D are stored [H, 1] and the gated norm [G, inner/G]; canonical weights are flat.</summary>
    public WeightInfo MapWeight(WeightInfo w, GgufMetadataView g)
    {
        if (w.Name.EndsWith(".ssm.a", StringComparison.Ordinal) || w.Name.EndsWith(".ssm.d", StringComparison.Ordinal) || w.Name.EndsWith(".ssm.norm.weight", StringComparison.Ordinal))
        {
            return w.As(w.Name, new Shape(w.Shape.Elements()));
        }

        return w;
    }
}
