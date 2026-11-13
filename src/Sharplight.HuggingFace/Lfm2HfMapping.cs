using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>lfm2</c> (Liquid LFM2 / LFM2.5: gated short-convolution layers and grouped-query attention layers with q/k norms) and
/// <c>lfm2_moe</c> (the same with sigmoid-routed experts and a selection bias after the leading dense layers) → canonical <c>hybrid</c>
/// (documented on <c>HybridAdapter</c>).
/// </summary>
internal sealed partial class Lfm2HfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "hybrid";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["lfm2", "lfm2_moe"];

    [GeneratedRegex(@"^feed_forward\.experts\.(\d+)\.(w1|w2|w3)\.weight$")]
    private static partial Regex ExpertName();

    /// <summary>Per-expert projection names to the canonical stacked weight (w1 gate, w3 up, w2 down).</summary>
    private static readonly Dictionary<string, string> s_expertProjection = new(StringComparer.Ordinal)
    {
        ["w1"] = "mlp.experts.gate.weight",
        ["w3"] = "mlp.experts.up.weight",
        ["w2"] = "mlp.experts.down.weight",
    };

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
            [a + "norm_eps"] = (float)(c.GetDouble("norm_eps", "block_norm_eps") ?? 1e-5),
            [a + "attention.head_count"] = heads,
            [a + "attention.kv_head_count"] = c.GetInt("num_key_value_heads") ?? heads,
            [a + "attention.head_dim"] = hidden / heads,
            [a + "attention.rope"] = true,
            [a + "rope.base"] = (float)(c.RopeDouble("rope_theta", "rope_theta") ?? 10000.0),
            [a + "shortconv.kernel"] = c.Int("conv_L_cache"),
        };
        if (c.GetBool("conv_bias") ?? false)
        {
            throw new NotSupportedException("lfm2: conv_bias is not supported.");
        }

        HfMappingHelpers.AddContextLength(md, a, c);
        if (c.GetInt("num_experts") is > 0 and int experts)
        {
            md[a + "moe.expert_count"] = experts;
            md[a + "moe.expert_used_count"] = c.Int("num_experts_per_tok");
            md[a + "moe.normalize_weights"] = c.GetBool("norm_topk_prob") ?? true;
            md[a + "moe.router_activation"] = "sigmoid";
            md[a + "moe.weights_scale"] = (float)(c.GetDouble("routed_scaling_factor") ?? 1.0);
        }

        return md;
    }

    [GeneratedRegex(@"^model\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    /// <summary>Per-layer tensors that are only renamed, HF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["operator_norm.weight"] = "input_norm.weight",
        ["ffn_norm.weight"] = "post_attention_norm.weight",
        ["conv.in_proj.weight"] = "shortconv.in.weight",
        ["conv.out_proj.weight"] = "shortconv.out.weight",
        ["self_attn.q_proj.weight"] = "attention.q.weight",
        ["self_attn.k_proj.weight"] = "attention.k.weight",
        ["self_attn.v_proj.weight"] = "attention.v.weight",
        ["self_attn.out_proj.weight"] = "attention.o.weight",
        ["self_attn.q_layernorm.weight"] = "attention.q_norm.weight",
        ["self_attn.k_layernorm.weight"] = "attention.k_norm.weight",
        ["feed_forward.w1.weight"] = "mlp.gate.weight",
        ["feed_forward.w3.weight"] = "mlp.up.weight",
        ["feed_forward.w2.weight"] = "mlp.down.weight",
    };

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var collector = new HfWeightCollector();
        var experts = HfPartJoiner.Experts();
        foreach (var (name, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            var w = tensor.ToWeightInfo();
            var m = LayerName().Match(name);
            bool mapped = m.Success ? TryMapLayerTensor($"layers.{m.Groups[1].Value}.", m.Groups[2].Value, w, collector, experts) : TryMapGlobalTensor(c, name, w, collector);
            if (!mapped)
            {
                collector.AddUnmapped(name);
            }
        }

        collector.ThrowIfUnmapped(c.ModelType);
        experts.JoinInto(collector, c.GetInt("num_experts") ?? 0);
        return collector.Weights;
    }

    private static bool TryMapGlobalTensor(HfConfig c, string name, WeightInfo w, HfWeightCollector collector)
    {
        switch (name)
        {
            case "model.embed_tokens.weight":
                collector.Add(HfTransforms.Plain(w, "token_embedding.weight"));
                return true;
            case "model.embedding_norm.weight":
                collector.Add(HfTransforms.Plain(w, "output_norm.weight"));   // applied to the final hidden states despite its name
                return true;
            case "lm_head.weight":
                if (!(c.GetBool("tie_word_embeddings", "tie_embedding") ?? false))
                {
                    collector.Add(HfTransforms.Plain(w, "lm_head.weight"));
                }

                return true;
            default:
                return false;
        }
    }

    private static bool TryMapLayerTensor(string prefix, string suffix, WeightInfo w, HfWeightCollector collector, HfPartJoiner experts)
    {
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

        switch (suffix)
        {
            case "feed_forward.gate.weight":
                collector.Add(HfTransforms.Plain(w, prefix + "mlp.router.weight"));
                return true;
            case "feed_forward.expert_bias":
                collector.Add(HfTransforms.ToF32(w, prefix + "mlp.router_bias"));
                return true;
        }

        if (suffix != "conv.conv.weight")
        {
            return false;
        }

        collector.Add(HfTransforms.ToF32(w, prefix + "shortconv.conv.weight", new Shape(w.Shape[0].Fixed, w.Shape[w.Shape.Rank - 1].Fixed)));
        return true;
    }
}
