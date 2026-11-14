using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>t5</c> (T5, T5 v1.1, Flan-T5, mT5) and <c>t5encoder</c> (their encoder alone) → canonical <c>encoder_decoder</c>
/// (vocabulary documented on <c>EncoderDecoderAdapter</c>). Encoder blocks are <c>enc.blk.N.*</c>, decoder blocks <c>dec.blk.N.*</c> with
/// <c>cross_attn_*</c> tensors; the bucketed relative position bias <c>attn_rel_b</c> of the first block serves the whole stack. Like
/// llama.cpp, gated feed-forward blocks use the tanh GELU and plain ones ReLU, the bias saturates at a distance of 128, and the LM head
/// is never rescaled (the tied-embedding scale of the original T5 changes the logits by a constant factor, not their order).
/// </summary>
internal sealed partial class T5GgufMapping : IGgufArchitectureMapping
{
    private const int MaxDistance = 128;

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["t5", "t5encoder"];

    /// <inheritdoc/>
    public string Architecture => "encoder_decoder";

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int layers = g.Int("block_count");
        int hidden = g.Int("embedding_length");
        int heads = g.Int("attention.head_count");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "encoder.layer_count"] = layers,
            [a + "decoder.layer_count"] = g.GgufArchitecture == "t5" ? g.Int("decoder_block_count", layers) : 0,
            [a + "attention.head_count"] = heads,
            [a + "attention.head_dim"] = g.Int("attention.key_length", hidden / heads),
            [a + "norm"] = "rms",
            [a + "norm_eps"] = g.Float("attention.layer_norm_rms_epsilon", g.Float("attention.layer_norm_epsilon", 1e-6f)),
            [a + "relative_attention.max_distance"] = MaxDistance,
            [a + "decoder_start_token"] = g.Int("decoder_start_token_id", 0),
            [a + "context_length"] = g.Int("context_length", 512),
        };
        if (GgufMappingHelpers.Pooling(g) is { } pooling)
        {
            md[CanonicalKeys.EmbeddingPooling] = pooling;
        }

        return md;
    }

    /// <summary>Gated feed-forward blocks (T5 v1.1, Flan-T5, mT5) use the tanh GELU, plain ones (the original T5) ReLU.</summary>
    public void RefineHyperParameters(Dictionary<string, object> hyperParameters, GgufMetadataView gguf, IReadOnlyDictionary<string, WeightInfo> weights)
        => hyperParameters[Architecture + ".mlp.activation"] = weights.ContainsKey("encoder.layers.0.mlp.gate.weight") ? "gelu_tanh" : "relu";

    [GeneratedRegex(@"^(enc|dec)\.blk\.(\d+)\.([a-z_]+)\.(weight|bias)$")]
    private static partial Regex StackBlock();

    /// <summary>Per-layer tensors, GGUF name (without the parameter suffix) to canonical module.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["attn_norm"] = "attention_norm",
        ["attn_q"] = "attention.q",
        ["attn_k"] = "attention.k",
        ["attn_v"] = "attention.v",
        ["attn_o"] = "attention.o",
        ["cross_attn_norm"] = "cross_attention_norm",
        ["cross_attn_q"] = "cross_attention.q",
        ["cross_attn_k"] = "cross_attention.k",
        ["cross_attn_v"] = "cross_attention.v",
        ["cross_attn_o"] = "cross_attention.o",
        ["ffn_norm"] = "mlp_norm",
        ["ffn_up"] = "mlp.up",
        ["ffn_gate"] = "mlp.gate",
        ["ffn_down"] = "mlp.down",
    };

    private static readonly Dictionary<string, string> s_global = new(StringComparer.Ordinal)
    {
        ["token_embd.weight"] = "token_embedding.weight",
        ["output.weight"] = "output.weight",
        ["enc.output_norm.weight"] = "encoder.output_norm.weight",
        ["dec.output_norm.weight"] = "decoder.output_norm.weight",
    };

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (s_global.TryGetValue(name, out var global))
        {
            return global;
        }

        var m = StackBlock().Match(name);
        if (!m.Success)
        {
            return null;
        }

        string stack = m.Groups[1].Value == "enc" ? "encoder" : "decoder";
        string layer = m.Groups[2].Value;
        string module = m.Groups[3].Value;
        string parameter = m.Groups[4].Value;
        if (module == "attn_rel_b")
        {
            // The first block's table serves the whole stack; a later block's (if any) is its own.
            return layer == "0" ? $"{stack}.attention.relative_bias.{parameter}" : $"{stack}.layers.{layer}.attention.relative_bias.{parameter}";
        }

        return s_layer.TryGetValue(module, out var canonical) ? $"{stack}.layers.{layer}.{canonical}.{parameter}" : null;
    }
}
