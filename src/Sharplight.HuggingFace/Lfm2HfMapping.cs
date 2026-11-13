using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>lfm2</c> (Liquid LFM2 / LFM2.5: gated short-convolution layers and grouped-query attention layers with q/k norms) → canonical <c>hybrid</c> (documented on <c>HybridAdapter</c>).</summary>
internal sealed partial class Lfm2HfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "hybrid";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["lfm2"];

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
        foreach (var (name, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            var w = tensor.ToWeightInfo();
            var m = LayerName().Match(name);
            bool mapped = m.Success ? TryMapLayerTensor($"layers.{m.Groups[1].Value}.", m.Groups[2].Value, w, collector) : TryMapGlobalTensor(c, name, w, collector);
            if (!mapped)
            {
                collector.AddUnmapped(name);
            }
        }

        collector.ThrowIfUnmapped("lfm2");
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

    private static bool TryMapLayerTensor(string prefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (s_layer.TryGetValue(suffix, out var renamed))
        {
            collector.Add(HfTransforms.Plain(w, prefix + renamed));
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
