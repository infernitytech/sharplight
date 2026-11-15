using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>paligemma</c> (PaliGemma, PaliGemma 2): the text decoder of the vision-language model, a Gemma (<c>text_config.model_type</c>
/// <c>gemma</c>) or Gemma 2 (<c>gemma2</c>) decoder mapped by that family's mapping. The language model's tensors sit below <c>language_model.</c>
/// (<c>language_model.model.*</c>, or <c>model.language_model.*</c> as transformers 5 saves them); the SigLIP tower and the projector are not
/// imported. The text configs omit <c>head_dim</c> and rely on the Gemma default of 256, which differs from hidden size over head count in
/// PaliGemma 2. Prompts are decoded causally: the bidirectional prefix (image and prompt tokens marked by the processor's <c>token_type_ids</c>)
/// is not modelled, so the outputs equal transformers' only for inputs without <c>token_type_ids</c> on its causal (eager) attention path; the
/// support level is therefore partial.
/// </summary>
internal sealed class PaliGemmaHfMapping : IHfArchitectureMapping
{
    private const int GemmaDefaultHeadDim = 256;
    private static readonly Gemma1HfMapping s_gemma = new();
    private static readonly DecoderHfMapping s_gemma2 = new();

    /// <inheritdoc/>
    public string Architecture => "decoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["paligemma"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig config)
    {
        if (TextModelType(config) is not ("gemma" or "gemma2"))
        {
            yield return $"paligemma: text model type '{TextModelType(config)}' is not supported (gemma, gemma2)";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string textType = TextModelType(c) ?? "gemma";
        using var text = c.AsModelType(textType);
        var md = new Dictionary<string, object>(TextMapping(textType).MapHyperParameters(text, options), StringComparer.Ordinal);
        if (c.GetInt("head_dim") is null)
        {
            md["decoder.attention.head_dim"] = GemmaDefaultHeadDim;
            md["decoder.rope.dim"] = GemmaDefaultHeadDim;
        }

        // transformers sets use_bidirectional_attention in PaliGemma's text config for its prefix; the decoder itself is causal, with the
        // configured (causal) sliding window.
        if (md.Remove("decoder.attention.causal") && c.GetInt("sliding_window") is { } window && md.ContainsKey("decoder.attention.sliding_window"))
        {
            md["decoder.attention.sliding_window"] = window;
        }

        return md;
    }

    /// <inheritdoc/>
    public string? PartialSupport(string modelType)
        => "text decoder only, decoded causally: the bidirectional prefix PaliGemma's processor marks (image and prompt tokens) is not modelled";

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        string textType = TextModelType(c) ?? "gemma";
        var textTensors = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (TextName(name) is { } textName)
            {
                textTensors[textName] = tensor;
            }
        }

        using var text = c.AsModelType(textType);
        return TextMapping(textType).MapWeights(text, textTensors, options);
    }

    private static IHfArchitectureMapping TextMapping(string textType) => textType == "gemma2" ? s_gemma2 : s_gemma;

    private static string? TextModelType(HfConfig c)
        => c.Root.TryGetProperty("text_config", out var text) && text.ValueKind == JsonValueKind.Object
           && text.TryGetProperty("model_type", out var type) && type.ValueKind == JsonValueKind.String
            ? type.GetString()
            : null;

    // The language model's tensors under the names a Gemma checkpoint uses; null for the vision tower and the projector.
    private static string? TextName(string name)
    {
        foreach (var (prefix, replacement) in s_prefixes)
        {
            if (name.StartsWith(prefix, StringComparison.Ordinal))
            {
                return replacement + name[prefix.Length..];
            }
        }

        return null;
    }

    private static readonly (string Prefix, string Replacement)[] s_prefixes =
    [
        ("language_model.model.", "model."),
        ("language_model.lm_head.", "lm_head."),
        ("model.language_model.", "model."),
        ("lm_head.", "lm_head."),
    ];
}
