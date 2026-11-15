using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// HunyuanOCR / Hunyuan-VL (HF <c>hunyuan_vl</c>, <c>hunyuan_vl_text</c>) → canonical <c>decoder</c>: the Hunyuan dense language model of
/// <see cref="HunyuanDenseHfMapping"/> (q/k norms after rope, NTK alpha folded into the base) with XD-RoPE: four sections
/// (<c>mrope_section</c>, the checkpoints' <c>xdrope_section</c>) that partition the head's rotated elements, so the two halves of a rotation pair
/// can follow different position axes (<c>rope.sections_per_element</c>), plus the vision tower and perceiver projector
/// (<see cref="HunyuanVlVisionHfMapping"/>) when the checkpoint has a <c>vision_config</c>. Hub checkpoints name the language model
/// <c>model.*</c>, transformers' modules <c>model.language_model.*</c>; the LM head is tied to the embedding.
/// </summary>
internal sealed class HunyuanVlHfMapping : IHfArchitectureMapping
{
    private const string LanguageModelPrefix = "model.language_model.";
    private readonly HunyuanDenseHfMapping _text = new();

    /// <inheritdoc/>
    public string Architecture => _text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["hunyuan_vl", "hunyuan_vl_text"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig config, HuggingFaceImportOptions options)
    {
        var md = new Dictionary<string, object>(_text.MapHyperParameters(config, options), StringComparer.Ordinal);
        if ((config.RopeIntArray("mrope_section") ?? config.RopeIntArray("xdrope_section")) is { } sections)
        {
            var axes = new int[4];
            for (int i = 0; i < Math.Min(axes.Length, sections.Length); i++)
            {
                axes[i] = sections[i];
            }

            md[Architecture + ".rope.sections"] = axes;
            md[Architecture + ".rope.sections_interleaved"] = false;
            md[Architecture + ".rope.sections_per_element"] = true;
        }

        HunyuanVlVisionHfMapping.AddVisionHyperParameters(config, md, Architecture, config.Int("hidden_size"));
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var text = tensors
            .Where(t => !HunyuanVlVisionHfMapping.IsVisionTensor(t.Key))
            .ToDictionary(
                t => t.Key.StartsWith(LanguageModelPrefix, StringComparison.Ordinal) ? "model." + t.Key[LanguageModelPrefix.Length..] : t.Key,
                t => t.Value,
                StringComparer.Ordinal);
        var weights = new Dictionary<string, WeightInfo>(_text.MapWeights(config, text, options), StringComparer.Ordinal);
        if (QwenVisionHfMapping.VisionConfig(config) is not null)
        {
            foreach (var (name, tensor) in tensors)
            {
                if (HunyuanVlVisionHfMapping.IsVisionTensor(name) && HunyuanVlVisionHfMapping.MapVisionTensor(name, tensor) is { } w)
                {
                    weights[w.Name] = w;
                }
            }
        }

        return weights;
    }
}
