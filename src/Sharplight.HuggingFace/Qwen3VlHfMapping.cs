using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Qwen3-VL (HF <c>qwen3_vl</c>) → canonical <c>decoder</c>, as a text-only model: the language model is Qwen3 with interleaved
/// M-RoPE and DeepStack, and for text tokens (every M-RoPE axis at the same position, no DeepStack features) that is exactly
/// the dense decoder with plain RoPE. The vision tower is skipped; image input is not supported yet.
/// </summary>
internal sealed class Qwen3VlHfMapping : IHfArchitectureMapping
{
    private const string LanguageModelPrefix = "model.language_model.";
    private const string VisionPrefix = "model.visual.";
    private readonly DecoderHfMapping _text = new();

    /// <inheritdoc/>
    public string Architecture => _text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["qwen3_vl", "qwen3_vl_text"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig config, HfImportOptions options) => _text.MapHyperParameters(config, options);

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        // model.language_model.X → model.X, the layout of the plain text model; the tower belongs to another source.
        var text = tensors
            .Where(t => !t.Key.StartsWith(VisionPrefix, StringComparison.Ordinal))
            .ToDictionary(
                t => t.Key.StartsWith(LanguageModelPrefix, StringComparison.Ordinal) ? "model." + t.Key[LanguageModelPrefix.Length..] : t.Key,
                t => t.Value,
                StringComparer.Ordinal);
        return _text.MapWeights(config, text, options);
    }
}
