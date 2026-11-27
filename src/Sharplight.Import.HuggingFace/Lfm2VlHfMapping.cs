using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// LFM2-VL (HF <c>lfm2_vl</c>, <c>Lfm2VlForConditionalGeneration</c>) → canonical <c>hybrid</c>: the LFM2 language model of
/// <see cref="Lfm2HfMapping"/> (tensors under <c>model.language_model.</c>, the <c>text_config</c>) plus the SigLIP 2 NaFlex tower and its
/// projector (<see cref="Lfm2VlVisionHfMapping"/>), the same canonical tower the GGUF <c>lfm2</c> mmproj gives.
/// </summary>
internal sealed class Lfm2VlHfMapping : IHfArchitectureMapping
{
    private const string LanguageModelPrefix = "model.language_model.";
    private readonly Lfm2HfMapping _text = new();

    /// <inheritdoc/>
    public string Architecture => _text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["lfm2_vl"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig config, HuggingFaceImportOptions options)
    {
        var md = new Dictionary<string, object>(_text.MapHyperParameters(config, options), StringComparer.Ordinal);
        Lfm2VlVisionHfMapping.AddVisionHyperParameters(config, md, Architecture, config.Int("hidden_size"));
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        // transformers 5 (and the Hub checkpoints) save model.language_model.X and lm_head; older exports have model.X directly.
        var text = tensors
            .Where(t => !Lfm2VlVisionHfMapping.IsVisionTensor(t.Key))
            .ToDictionary(
                t => t.Key.StartsWith(LanguageModelPrefix, StringComparison.Ordinal) ? "model." + t.Key[LanguageModelPrefix.Length..] : t.Key,
                t => t.Value,
                StringComparer.Ordinal);
        var weights = new Dictionary<string, WeightInfo>(_text.MapWeights(config, text, options), StringComparer.Ordinal);
        var unmapped = new List<string>();
        int channels = Lfm2VlVisionHfMapping.Channels(config);
        foreach (var (name, tensor) in tensors)
        {
            if (!Lfm2VlVisionHfMapping.IsVisionTensor(name) || Lfm2VlVisionHfMapping.IsUnusedTensor(name))
            {
                continue;
            }

            if (Lfm2VlVisionHfMapping.MapVisionTensor(name, tensor, channels) is { } w)
            {
                weights[w.Name] = w;
            }
            else
            {
                unmapped.Add(name);
            }
        }

        if (unmapped.Count > 0)
        {
            throw new NotSupportedException($"lfm2_vl: unmapped vision tensors: {string.Join(", ", unmapped.Take(8))}{(unmapped.Count > 8 ? ", ..." : "")}.");
        }

        return weights;
    }
}
