using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>The hyper-parameters every llama.cpp decoder-only architecture shares, as canonical <c>decoder.*</c> keys.</summary>
internal static class GgufDecoderHyperParameters
{
    /// <summary>Prefix of the canonical decoder keys.</summary>
    public const string Prefix = "decoder.";

    /// <summary>
    /// Layer / hidden / feed-forward sizes, head counts and head size, rotary dimension, base and pairing, norm epsilon, context length
    /// and the MLP activation. The norm epsilon comes from the RMS key, or from the layer-norm key for layer-norm models. A file that declares
    /// <c>attention.causal</c> = false (bidirectional embedding models) gets canonical <c>attention.causal</c> = false, and llama.cpp's
    /// <c>pooling_type</c> (mean, cls, last; rank pools the cls row) becomes <see cref="CanonicalKeys.EmbeddingPooling"/>.
    /// </summary>
    /// <param name="g">The GGUF metadata.</param>
    /// <param name="activation">Canonical activation name.</param>
    /// <param name="ropeStyle">Rotary pairing the rope op uses on the canonical q/k rows ("neox" or "gptj").</param>
    /// <param name="layerNorm">Whether the model normalizes with mean-centred layer norm instead of RMS norm.</param>
    public static Dictionary<string, object> Common(GgufMetadataView g, string activation, string ropeStyle, bool layerNorm)
    {
        int embeddingLength = g.Int("embedding_length");
        int headCount = LargestOf(g, "attention.head_count");
        int headDim = g.Int("attention.key_length", embeddingLength / headCount);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [Prefix + "layer_count"] = g.Int("block_count"),
            [Prefix + "hidden_size"] = embeddingLength,
            [Prefix + "ffn_hidden_size"] = LargestOf(g, "feed_forward_length", g.Int("expert_feed_forward_length", 0)),
            [Prefix + "norm_eps"] = g.Has("attention.layer_norm_rms_epsilon") ? g.Float("attention.layer_norm_rms_epsilon") : g.Float("attention.layer_norm_epsilon"),
            [Prefix + "attention.head_count"] = headCount,
            [Prefix + "attention.kv_head_count"] = LargestOf(g, "attention.head_count_kv", headCount),
            [Prefix + "attention.head_dim"] = headDim,
            [Prefix + "rope.dim"] = g.Int("rope.dimension_count", headDim),
            [Prefix + "rope.base"] = g.Float("rope.freq_base", GgufMappingHelpers.DefaultRopeBase),
            [Prefix + "rope.style"] = ropeStyle,
            [Prefix + "mlp.activation"] = activation,
        };
        if (layerNorm)
        {
            md[Prefix + "norm.type"] = "layer";
        }

        GgufMappingHelpers.AddContextLength(md, Prefix, g);
        if (!g.Bool("attention.causal", true))
        {
            md[Prefix + "attention.causal"] = false;
        }

        if (GgufMappingHelpers.Pooling(g) is { } pooling and not "none")
        {
            md[CanonicalKeys.EmbeddingPooling] = pooling;
        }

        return md;
    }

    /// <summary>An integer key that may be stored per layer (an array, e.g. OpenELM): the largest entry; <paramref name="fallback"/> when absent.</summary>
    public static int LargestOf(GgufMetadataView g, string key, int? fallback = null)
    {
        if (g.Raw(key) is Array values)
        {
            return values.Cast<object>().Max(Convert.ToInt32);
        }

        return fallback is { } value ? g.Int(key, value) : g.Int(key);
    }
}
