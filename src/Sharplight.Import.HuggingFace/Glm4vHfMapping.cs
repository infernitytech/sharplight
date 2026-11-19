using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// GLM-4.1V and GLM-OCR (HF <c>glm4v</c>, <c>glm4v_text</c>, <c>glm_ocr</c>, <c>glm_ocr_text</c>; the OCR model's multi-token-prediction layer is not read) → canonical <c>decoder</c>: the GLM-4 language model of <see cref="Glm4HfMapping"/> (q/k/v
/// biases, sandwich norms, rotary on interleaved pairs of the first <c>partial_rotary_factor</c> of each head) with sectioned, non-interleaved
/// M-RoPE (<c>rope.sections</c> from <c>mrope_section</c>), plus the vision tower (<see cref="Glm4vVisionHfMapping"/>) when the checkpoint has a
/// <c>vision_config</c>. Image prompts follow the Qwen-VL layout (<c>Qwen35Multimodal</c>) with <c>&lt;|image|&gt;</c> as the placeholder.
/// </summary>
internal sealed class Glm4vHfMapping : IHfArchitectureMapping
{
    private const string LanguageModelPrefix = "model.language_model.";
    private readonly Glm4HfMapping _text = new();

    /// <inheritdoc/>
    public string Architecture => _text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["glm4v", "glm4v_text", "glm_ocr", "glm_ocr_text"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig config, HuggingFaceImportOptions options)
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

        Glm4vVisionHfMapping.AddVisionHyperParameters(config, md, Architecture);
        AddEndOfTurn(config, md);
        if (config.ModelType is "glm_ocr" or "glm_ocr_text")
        {
            // The prompts of the model card's recognition tasks (the image comes first in the user turn).
            md[CanonicalKeys.RecognitionTasks] = new[] { "text=Text Recognition:", "formula=Formula Recognition:", "table=Table Recognition:" };
        }

        return md;
    }

    // GLM-OCR stops at <|endoftext|> and at <|user|> (eos_token_id lists both); the tokenizer's eos is <|endoftext|>, the other id ends the turn.
    private static void AddEndOfTurn(HfConfig config, Dictionary<string, object> md)
    {
        if (config.Get("eos_token_id") is not { ValueKind: JsonValueKind.Array } ids)
        {
            return;
        }

        var all = ids.EnumerateArray().Select(e => e.GetInt32()).ToArray();
        if (all.Length > 1)
        {
            md[CanonicalKeys.TokenizerEotId] = all[^1];
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        // transformers 5 saves model.language_model.X and model.visual.*; older checkpoints model.X and model.visual.*.
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
                if (QwenVisionHfMapping.IsVisionTensor(name) && Glm4vVisionHfMapping.MapVisionTensor(config, name, tensor) is { } w)
                {
                    weights[w.Name] = w;
                }
            }
        }

        return weights;
    }
}
