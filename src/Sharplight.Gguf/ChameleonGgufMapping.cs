using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>chameleon</c>: Llama layers (q/k rows permuted for llama.cpp's interleaved rope) with a layer norm over each q and k head, one
/// gain and bias per head ([heads, head_dim], also stored in interleaved order and un-permuted here), epsilon 1e-5. With <c>swin_norm</c> the
/// two layer norms follow the attention and MLP outputs instead of preceding them. llama.cpp suppresses the image tokens [4, 8196) of the
/// Chameleon vocabulary in the logits; the same range becomes the canonical <c>logits.suppressed_ranges</c>.
/// </summary>
internal sealed class ChameleonGgufMapping : GgufDecoderFamilyMapping, IGgufWeightTransformer
{
    // llama.cpp's graph masks this fixed range of image-token logits (it does not read the range from the file).
    private const int FirstImageToken = 4;
    private const int EndImageToken = 8196;

    private static readonly Dictionary<string, string> s_tensors = new(StringComparer.Ordinal)
    {
        ["attn_q_norm.bias"] = "attention.q_norm.bias",
        ["attn_k_norm.bias"] = "attention.k_norm.bias",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["chameleon"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_tensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        const string a = GgufDecoderHyperParameters.Prefix;
        md[a + "attention.qk_norm_type"] = "layer";
        md[a + "attention.qk_norm_eps"] = 1e-5f;
        if (SwinNorm(g))
        {
            md[a + "pre_norm"] = false;
        }

        int vocab = g.Int("vocab_size", 0);
        if (vocab == 0 || vocab > FirstImageToken)
        {
            md[a + "logits.suppressed_ranges"] = new[] { FirstImageToken, EndImageToken };
        }
    }

    private static bool SwinNorm(GgufMetadataView g) => g.Raw("swin_norm") is true;

    /// <summary>Un-permutes the per-head q/k norm gains and biases; with swin norm, renames the layer norms to the sub-layer output norms.</summary>
    public void Transform(Dictionary<string, WeightInfo> weights, GgufMetadataView gguf)
    {
        bool swin = SwinNorm(gguf);
        foreach (var (name, w) in weights.ToArray())
        {
            if (name.EndsWith("attention.q_norm.weight", StringComparison.Ordinal) || name.EndsWith("attention.q_norm.bias", StringComparison.Ordinal)
                || name.EndsWith("attention.k_norm.weight", StringComparison.Ordinal) || name.EndsWith("attention.k_norm.bias", StringComparison.Ordinal))
            {
                weights[name] = UnpermuteHeads(w);
            }
            else if (swin && name.EndsWith("input_norm.weight", StringComparison.Ordinal))
            {
                weights.Remove(name);
                weights[name[..^"input_norm.weight".Length] + "attention.out_norm.weight"] = w.As(name[..^"input_norm.weight".Length] + "attention.out_norm.weight");
            }
            else if (swin && name.EndsWith("post_attention_norm.weight", StringComparison.Ordinal))
            {
                weights.Remove(name);
                weights[name[..^"post_attention_norm.weight".Length] + "mlp.out_norm.weight"] = w.As(name[..^"post_attention_norm.weight".Length] + "mlp.out_norm.weight");
            }
        }
    }

    // A [heads, head_dim] norm parameter in interleaved order back to rotate-half order within every head.
    private static WeightInfo UnpermuteHeads(WeightInfo w)
    {
        if (w.Shape.Rank != 2)
        {
            return w;
        }

        int heads = checked((int)w.Shape[0].Fixed);
        var flat = w.As(w.Name, new Shape(w.Shape.Elements()));
        return DecoderGgufMapping.UnpermuteRows(flat, heads).As(w.Name, w.Shape);
    }
}
