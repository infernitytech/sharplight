using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Qwen3-VL (HF <c>qwen3_vl</c>, <c>qwen3_vl_text</c>) → canonical <c>decoder</c>: the language model is Qwen3 with interleaved M-RoPE
/// (<c>rope.sections</c> from <c>mrope_section</c>, padded to 4 axes) and the vision tower (with its DeepStack mergers) is mapped by
/// <see cref="QwenVisionHfMapping"/> into the same source.
/// </summary>
internal sealed class Qwen3VlHfMapping : IHfArchitectureMapping
{
    private const string LanguageModelPrefix = "model.language_model.";
    private readonly DecoderHfMapping _text = new();

    /// <inheritdoc/>
    public string Architecture => _text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["qwen3_vl", "qwen3_vl_text"];

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
            md[Architecture + ".rope.sections_interleaved"] = config.GlobalRopeParameters is { } rp && rp.TryGetProperty("mrope_interleaved", out var il)
                ? il.ValueKind == System.Text.Json.JsonValueKind.True : true;
        }

        QwenVisionHfMapping.AddVisionHyperParameters(config, md, Architecture);
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        // model.language_model.X → model.X, the layout of the plain text model; the tower is mapped separately.
        var text = tensors
            .Where(t => !QwenVisionHfMapping.IsVisionTensor(t.Key))
            .ToDictionary(
                t => t.Key.StartsWith(LanguageModelPrefix, StringComparison.Ordinal) ? "model." + t.Key[LanguageModelPrefix.Length..] : t.Key,
                t => t.Value,
                StringComparer.Ordinal);
        var weights = new Dictionary<string, WeightInfo>(_text.MapWeights(config, text, options), StringComparer.Ordinal);
        if (QwenVisionHfMapping.VisionConfig(config) is not null)
        {
            foreach (var (name, tensor) in tensors)
            {
                if (QwenVisionHfMapping.IsVisionTensor(name) && QwenVisionHfMapping.MapVisionTensor(config, name, tensor) is { } w)
                {
                    weights[w.Name] = w;
                }
            }
        }

        return weights;
    }
}
