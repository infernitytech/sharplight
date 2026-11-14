using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// PaddleOCR-VL language model (HF <c>paddleocr_vl</c>, <c>paddleocr_vl_text</c>) → canonical <c>decoder</c>: ERNIE-4.5 (optional biases)
/// with sectioned, non-interleaved M-RoPE (<c>rope.sections</c> from <c>mrope_section</c>, padded to 4 axes), like llama.cpp's
/// <c>paddleocr</c>. The SigLIP-style tower (<c>visual.*</c>) and its projector (<c>mlp_AR.*</c>) are not imported.
/// </summary>
internal sealed class PaddleOcrVlHfMapping : IHfArchitectureMapping
{
    private const string LanguageModelPrefix = "model.language_model.";
    private readonly DecoderHfMapping _text = new();

    /// <inheritdoc/>
    public string Architecture => _text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["paddleocr_vl", "paddleocr_vl_text"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig config, HfImportOptions options)
    {
        var md = new Dictionary<string, object>(_text.MapHyperParameters(config, options), StringComparer.Ordinal);
        if (config.RopeIntArray("mrope_section") is { } sections)
        {
            var axes = new int[4];
            for (int i = 0; i < Math.Min(axes.Length, sections.Length); i++)
            {
                axes[i] = sections[i];
            }

            md[Architecture + ".rope.sections"] = axes;
            md[Architecture + ".rope.sections_interleaved"] = false;
        }

        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        // Hub checkpoints have model.X / lm_head / visual.* / mlp_AR.*; transformers 5 saves model.language_model.X, model.visual.*, model.projector.*.
        var text = tensors
            .Where(t => !IsImageTensor(t.Key))
            .ToDictionary(
                t => t.Key.StartsWith(LanguageModelPrefix, StringComparison.Ordinal) ? "model." + t.Key[LanguageModelPrefix.Length..] : t.Key,
                t => t.Value,
                StringComparer.Ordinal);
        return _text.MapWeights(config, text, options);
    }

    private static bool IsImageTensor(string name)
        => name.StartsWith("visual.", StringComparison.Ordinal) || name.StartsWith("model.visual.", StringComparison.Ordinal)
        || name.StartsWith("mlp_AR.", StringComparison.Ordinal) || name.StartsWith("model.projector.", StringComparison.Ordinal);
}
