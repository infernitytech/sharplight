using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Qwen2-VL and Qwen2.5-VL language models (HF <c>qwen2_vl</c>, <c>qwen2_5_vl</c> and their <c>_text</c> configs) → canonical <c>decoder</c>:
/// Qwen2 (q/k/v biases) with sectioned, non-interleaved M-RoPE (<c>rope.sections</c> from <c>mrope_section</c>, padded to 4 axes). Only the
/// language model is imported: the vision tower (<c>visual.*</c>, 2-D rotary patches, windowed attention in 2.5) is not mapped, so the
/// model runs text-only, where every position axis is equal and M-RoPE is plain RoPE.
/// </summary>
internal sealed class Qwen2VlHfMapping : IHfArchitectureMapping
{
    private const string LanguageModelPrefix = "model.language_model.";
    private readonly DecoderHfMapping _text = new();

    /// <inheritdoc/>
    public string Architecture => _text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["qwen2_vl", "qwen2_vl_text", "qwen2_5_vl", "qwen2_5_vl_text"];

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
        // model.language_model.X (transformers 5) → model.X; the older layout already has model.X. The tower is left out.
        var text = tensors
            .Where(t => !QwenVisionHfMapping.IsVisionTensor(t.Key))
            .ToDictionary(
                t => t.Key.StartsWith(LanguageModelPrefix, StringComparison.Ordinal) ? "model." + t.Key[LanguageModelPrefix.Length..] : t.Key,
                t => t.Value,
                StringComparer.Ordinal);
        return _text.MapWeights(config, text, options);
    }
}
