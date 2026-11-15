using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face <c>llava_onevision2</c> (LLaVA-OneVision-2 remote code, e.g. Mage-VL): the text decoder of the vision-language model, a decoder
/// family of <see cref="DecoderHfMapping"/> (<c>text_config.model_type</c>, Qwen3 in the released checkpoints) mapped by that mapping. Text positions
/// are plain 1D positions, so the text decoder is that family's unchanged; the vision tower (<c>visual.*</c>) is not imported. The language model's
/// tensors are stored as <c>model.*</c> (or <c>model.language_model.*</c> as transformers 5 saves them).
/// </summary>
internal sealed class LlavaOneVision2HfMapping : IHfArchitectureMapping
{
    private const string VisionPrefix = "visual.";
    private const string NestedTextPrefix = "model.language_model.";
    private static readonly DecoderHfMapping s_text = new();

    /// <inheritdoc/>
    public string Architecture => s_text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["llava_onevision2"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig config)
    {
        string? textType = TextModelType(config);
        if (textType is null || !s_text.ModelTypes.Contains(textType))
        {
            yield return $"llava_onevision2: text model type '{textType}' is not supported";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        using var text = c.AsModelType(TextModelType(c)!);
        return s_text.MapHyperParameters(text, options);
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var textTensors = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (name.StartsWith(VisionPrefix, StringComparison.Ordinal) || name.StartsWith("model." + VisionPrefix, StringComparison.Ordinal))
            {
                continue;
            }

            textTensors[name.StartsWith(NestedTextPrefix, StringComparison.Ordinal) ? "model." + name[NestedTextPrefix.Length..] : name] = tensor;
        }

        using var text = c.AsModelType(TextModelType(c)!);
        return s_text.MapWeights(text, textTensors, options);
    }

    private static string? TextModelType(HfConfig c)
        => c.Root.TryGetProperty("text_config", out var text) && text.ValueKind == JsonValueKind.Object
           && text.TryGetProperty("model_type", out var type) && type.ValueKind == JsonValueKind.String
            ? type.GetString()
            : null;
}
