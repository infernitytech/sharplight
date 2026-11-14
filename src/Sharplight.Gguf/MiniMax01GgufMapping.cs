using System.Runtime.InteropServices;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>minimax-01</c> (MiniMax-Text-01 / MiniMax-M1: lightning-attention layers interleaved with grouped-query attention, residuals from
/// the normalized input scaled by <c>residual_scale</c>, softmax-routed experts) → canonical <c>hybrid</c> (documented on <c>HybridAdapter</c>).
/// Lightning layers are flagged by <c>attention.recurrent_layers</c> (else every layer but each <c>full_attention_interval</c>-th, default 8).
/// The files carry no decay: the per-head slopes are computed as llama.cpp does, start^(h+1) with start = 2^(−2^(−(log2 heads − 3))), scaled by
/// 1 − layer / (layers − 1) + 1e-5, and stored as the canonical log decay −slope.
/// </summary>
internal sealed class MiniMax01GgufMapping : IGgufArchitectureMapping
{
    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["minimax-01"];

    /// <inheritdoc/>
    public string Architecture => "hybrid";

    /// <summary>Per-layer tensors of softmax-attention layers, GGUF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_attention = new(StringComparer.Ordinal)
    {
        ["attn_q.weight"] = "attention.q.weight",
        ["attn_k.weight"] = "attention.k.weight",
        ["attn_v.weight"] = "attention.v.weight",
        ["attn_output.weight"] = "attention.o.weight",
    };

    /// <summary>Per-layer tensors of lightning-attention layers.</summary>
    private static readonly Dictionary<string, string> s_lightning = new(StringComparer.Ordinal)
    {
        ["attn_qkv.weight"] = "lightning.qkv.weight",
        ["attn_norm_2.weight"] = "lightning.norm.weight",
        ["attn_gate.weight"] = "lightning.gate.weight",
        ["attn_output.weight"] = "lightning.out.weight",
    };

    /// <summary>Per-layer tensors of every layer.</summary>
    private static readonly Dictionary<string, string> s_common = new(StringComparer.Ordinal)
    {
        ["attn_norm.weight"] = "input_norm.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["ffn_gate_inp.weight"] = "mlp.router.weight",
        ["ffn_gate_exps.weight"] = "mlp.experts.gate.weight",
        ["ffn_up_exps.weight"] = "mlp.experts.up.weight",
        ["ffn_down_exps.weight"] = "mlp.experts.down.weight",
    };

    private static bool IsLightningLayer(GgufMetadataView g, int layer)
    {
        if (g.Raw("attention.recurrent_layers") is Array flags && layer < flags.Length)
        {
            return Convert.ToBoolean(flags.GetValue(layer));
        }

        return (layer + 1) % g.Int("full_attention_interval", 8) != 0;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int layers = g.Int("block_count");
        int heads = g.Int("attention.head_count");
        int headDim = g.Int("attention.key_length", g.Int("embedding_length") / heads);
        var scale = Enumerable.Repeat(g.Float("residual_scale", 1f), layers).ToArray();
        var ones = Enumerable.Repeat(1f, layers).ToArray();
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layers,
            [a + "hidden_size"] = g.Int("embedding_length"),
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon"),
            [a + "attention.head_count"] = heads,
            [a + "attention.kv_head_count"] = g.Int("attention.head_count_kv", heads),
            [a + "attention.head_dim"] = headDim,
            [a + "attention.rope"] = true,
            [a + "rope.dim"] = g.Int("rope.dimension_count", headDim),
            [a + "rope.base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase),
            [a + "lightning.head_count"] = heads,
            [a + "lightning.head_dim"] = headDim,
            [a + "residual.from_norm"] = true,
            [a + "residual.mixer_input_scale"] = scale,
            [a + "residual.mixer_scale"] = ones,
            [a + "residual.ffn_input_scale"] = scale,
            [a + "residual.ffn_scale"] = ones,
            [a + "moe.expert_count"] = g.Int("expert_count"),
            [a + "moe.expert_used_count"] = g.Int("expert_used_count"),
            [a + "moe.normalize_weights"] = true,
            [a + "moe.router_activation"] = "softmax",
        };
        GgufMappingHelpers.AddContextLength(md, a, g);
        return md;
    }

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (GgufTensorNames.MapGlobal(name) is { } globalName)
        {
            return globalName;
        }

        var m = GgufTensorNames.Block().Match(name);
        if (!m.Success)
        {
            return null;
        }

        int layer = int.Parse(m.Groups[1].Value);
        string suffix = m.Groups[2].Value;
        var kind = IsLightningLayer(g, layer) ? s_lightning : s_attention;
        return kind.TryGetValue(suffix, out var canonical) || s_common.TryGetValue(suffix, out canonical) ? $"layers.{layer}.{canonical}" : null;
    }

    /// <inheritdoc/>
    public IEnumerable<WeightInfo> SyntheticWeights(GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        int layers = g.Int("block_count");
        int heads = g.Int("attention.head_count");
        float start = MathF.Pow(2f, -MathF.Pow(2f, -(MathF.Log2(heads) - 3f)));
        for (int il = 0; il < layers; il++)
        {
            if (!IsLightningLayer(g, il))
            {
                continue;
            }

            float factor = 1f - (float)il / (layers - 1) + 1e-5f;
            var decay = Enumerable.Range(1, heads).Select(e => -MathF.Pow(start, e) * factor).ToArray();
            yield return WeightInfo.Derived($"layers.{il}.lightning.decay", DataType.F32, new Shape(heads),
                () => MemoryMarshal.AsBytes(decay.AsSpan()).ToArray());
        }
    }
}
