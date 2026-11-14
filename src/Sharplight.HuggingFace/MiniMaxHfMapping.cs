using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>minimax</c> (MiniMax-01 / MiniMax-Text-01: lightning-attention layers interleaved with grouped-query attention, residuals taken
/// from the normalized input with per-kind (α, β) scales, softmax-routed experts) → canonical <c>hybrid</c> (documented on <c>HybridAdapter</c>).
/// </summary>
/// <remarks>
/// The log decay of a lightning layer is minus its <c>slope_rate</c> buffer when the checkpoint has one, else computed as the reference does:
/// base^(h+1) · (1 − layer / (layers − 1 + 1e-5) + 1e-5) with base = 2^(−8 / heads). The rotary embedding covers <c>rotary_dim</c> channels when
/// the config names it (the original model and llama.cpp); transformers rotates the whole head.
/// </remarks>
internal sealed partial class MiniMaxHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "hybrid";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["minimax"];

    [GeneratedRegex(@"^block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$")]
    private static partial Regex ExpertName();

    [GeneratedRegex(@"^model\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    /// <summary>Per-expert projection names to the canonical stacked weight (w1 gate, w3 up, w2 down).</summary>
    private static readonly Dictionary<string, string> s_expertProjection = new(StringComparer.Ordinal)
    {
        ["w1"] = "mlp.experts.gate.weight",
        ["w3"] = "mlp.experts.up.weight",
        ["w2"] = "mlp.experts.down.weight",
    };

    /// <summary>Per-layer tensors that are only renamed, HF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["input_layernorm.weight"] = "input_norm.weight",
        ["post_attention_layernorm.weight"] = "post_attention_norm.weight",
        ["self_attn.q_proj.weight"] = "attention.q.weight",
        ["self_attn.k_proj.weight"] = "attention.k.weight",
        ["self_attn.v_proj.weight"] = "attention.v.weight",
        ["self_attn.o_proj.weight"] = "attention.o.weight",
        ["self_attn.qkv_proj.weight"] = "lightning.qkv.weight",
        ["self_attn.norm.weight"] = "lightning.norm.weight",
        ["self_attn.output_gate.weight"] = "lightning.gate.weight",
        ["self_attn.out_proj.weight"] = "lightning.out.weight",
        ["block_sparse_moe.gate.weight"] = "mlp.router.weight",
    };

    /// <summary>Buffers the reference derives from the slope rate; the graph does not need them.</summary>
    private static readonly HashSet<string> s_derivedBuffers = new(StringComparer.Ordinal)
    {
        "self_attn.query_decay", "self_attn.key_decay", "self_attn.diagonal_decay",
    };

    private static int Heads(HfConfig c) => c.Int("num_attention_heads");

    private static int HeadDim(HfConfig c) => c.GetInt("head_dim") ?? c.Int("hidden_size") / Heads(c);

    /// <summary>Whether layer <paramref name="layer"/> is a lightning-attention layer (<c>layer_types</c>, else the original <c>attn_type_list</c>, 0 = linear).</summary>
    private static bool IsLinearLayer(HfConfig c, int layer)
    {
        if (c.GetStringArray("layer_types") is { } types && layer < types.Length)
        {
            return types[layer] == "linear_attention";
        }

        if (c.Get("attn_type_list") is { ValueKind: JsonValueKind.Array } list && layer < list.GetArrayLength())
        {
            return list[layer].GetInt32() == 0;
        }

        return (layer + 1) % 8 != 0;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        int layers = c.Int("num_hidden_layers");
        int heads = Heads(c);
        int headDim = HeadDim(c);
        var linear = Enumerable.Range(0, layers).Select(il => IsLinearLayer(c, il)).ToArray();
        float Factor(string key) => (float)(c.GetDouble(key) ?? 1.0);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = layers,
            [a + "hidden_size"] = c.Int("hidden_size"),
            [a + "norm_eps"] = (float)(c.GetDouble("rms_norm_eps") ?? 1e-5),
            [a + "attention.head_count"] = heads,
            [a + "attention.kv_head_count"] = c.GetInt("num_key_value_heads") ?? heads,
            [a + "attention.head_dim"] = headDim,
            [a + "attention.rope"] = true,
            [a + "rope.base"] = (float)(c.RopeDouble("rope_theta", "rope_theta") ?? 10000.0),
            [a + "rope.dim"] = c.GetInt("rotary_dim") ?? headDim,
            [a + "lightning.head_count"] = heads,
            [a + "lightning.head_dim"] = headDim,
            [a + "residual.from_norm"] = true,
            [a + "residual.mixer_input_scale"] = linear.Select(l => Factor(l ? "linear_attn_alpha_factor" : "full_attn_alpha_factor")).ToArray(),
            [a + "residual.mixer_scale"] = linear.Select(l => Factor(l ? "linear_attn_beta_factor" : "full_attn_beta_factor")).ToArray(),
            [a + "residual.ffn_input_scale"] = Enumerable.Repeat(Factor("mlp_alpha_factor"), layers).ToArray(),
            [a + "residual.ffn_scale"] = Enumerable.Repeat(Factor("mlp_beta_factor"), layers).ToArray(),
            [a + "moe.expert_count"] = c.Int("num_local_experts"),
            [a + "moe.expert_used_count"] = c.Int("num_experts_per_tok"),
            [a + "moe.normalize_weights"] = true,
            [a + "moe.router_activation"] = "softmax",
        };
        HfMappingHelpers.AddContextLength(md, a, c);
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var collector = new HfWeightCollector();
        var experts = HfPartJoiner.Experts();
        var slopes = new HashSet<int>();
        foreach (var (name, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            var w = tensor.ToWeightInfo();
            var m = LayerName().Match(name);
            bool mapped = m.Success
                ? TryMapLayerTensor(c, int.Parse(m.Groups[1].Value), m.Groups[2].Value, w, collector, experts, slopes)
                : TryMapGlobalTensor(c, name, w, collector);
            if (!mapped)
            {
                collector.AddUnmapped(name);
            }
        }

        collector.ThrowIfUnmapped(c.ModelType);
        experts.JoinInto(collector, c.Int("num_local_experts"));
        int layers = c.Int("num_hidden_layers");
        for (int il = 0; il < layers; il++)
        {
            if (IsLinearLayer(c, il) && !slopes.Contains(il))
            {
                var decay = ComputedDecay(Heads(c), il, layers);
                collector.Add(WeightInfo.Derived($"layers.{il}.lightning.decay", DataType.F32, new Shape(decay.Length), () => HfTransforms.EncodeF32(decay)));
            }
        }

        return collector.Weights;
    }

    /// <summary>The log decay −slope of every head of lightning layer <paramref name="layer"/>, as the reference computes it.</summary>
    private static float[] ComputedDecay(int heads, int layer, int layers)
    {
        float baseRate = 1f / MathF.Pow(2f, 8f / heads);
        float factor = 1f - layer / (layers - 1 + 1e-5f) + 1e-5f;
        return [.. Enumerable.Range(1, heads).Select(e => -MathF.Pow(baseRate, e) * factor)];
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

    private static bool TryMapLayerTensor(HfConfig c, int layer, string suffix, WeightInfo w, HfWeightCollector collector, HfPartJoiner experts,
        HashSet<int> slopes)
    {
        string prefix = $"layers.{layer}.";
        if (s_layer.TryGetValue(suffix, out var renamed))
        {
            collector.Add(HfTransforms.Plain(w, prefix + renamed));
            return true;
        }

        if (ExpertName().Match(suffix) is { Success: true } expert)
        {
            experts.Add(prefix + s_expertProjection[expert.Groups[2].Value], int.Parse(expert.Groups[1].Value), w);
            return true;
        }

        if (suffix == "self_attn.slope_rate")
        {
            collector.Add(HfTransforms.ToF32(w, prefix + "lightning.decay", new Shape(Heads(c)), static v => -v));
            slopes.Add(layer);
            return true;
        }

        return s_derivedBuffers.Contains(suffix);
    }
}
