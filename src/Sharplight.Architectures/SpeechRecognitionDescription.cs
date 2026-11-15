using System.Globalization;
using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the <see cref="ModelDescription"/> of an encoder-decoder speech recognizer (kind <see cref="ModelKinds.SpeechRecognition"/>):
/// <c>encoder</c> and <c>decoder</c> components over the model's graphs, a <c>tokenizer</c> component whose tokenizer is stored as an
/// attachment (<see cref="ModelComponentAttachments.Tokenizer"/>, the canonical <see cref="TokenizerData"/> JSON) and a log-mel
/// <c>feature-extractor</c> component configured from the canonical <c>{arch}.audio.*</c> hyper-parameters with its filter bank attachment.
/// The tokenizer keys leave the definition's metadata, so the package stores the vocabulary once.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>{arch}.audio.</c>): <c>sample_rate</c>, <c>fft_size</c>, <c>hop_length</c>, <c>mel_bins</c>, <c>chunk_samples</c> (the
/// encoder window, also appended as silence so every window is complete), optional <c>log_floor</c> (default 1e-10),
/// <c>dynamic_range</c> (log10 units below the recording's peak, default 8), <c>log_offset</c> (default 4) and <c>log_scale</c> (default
/// 0.25): Whisper's normalization. Source attachment <see cref="MelFiltersAttachment"/>: f32 [mel_bins, fft_size / 2 + 1].
/// </remarks>
internal static class SpeechRecognitionDescription
{
    /// <summary>The canonical source attachment of the mel filter bank.</summary>
    public const string MelFiltersAttachment = "feature-extractor/mel-filters";

    /// <summary>The attachment that stores the tokenizer.</summary>
    public const string TokenizerAttachment = "tokenizer/tokenizer.json";

    /// <summary>The symbol of the decoder's logits rows (the speech decoder reads the last row only).</summary>
    private const string OutputRowsSymbol = "R";

    /// <summary>The description, the definition's attachments and its metadata without the tokenizer keys.</summary>
    /// <param name="src">The model source (hyper-parameters, tokenizer metadata, filter bank attachment).</param>
    /// <param name="decoderContext">The decoder's context length in tokens.</param>
    /// <param name="sourceRows">Encoder output rows per window.</param>
    /// <exception cref="NotSupportedException">The source has no filter bank or no tokenizer.</exception>
    public static (ModelDescription Description, IReadOnlyDictionary<string, IWeightSource> Attachments, IReadOnlyDictionary<string, object> Metadata) Build(
        ModelSource src, int decoderContext, int sourceRows)
    {
        if (!src.Attachments.ContainsKey(MelFiltersAttachment))
        {
            throw new NotSupportedException($"{src.Architecture}: a speech encoder needs the '{MelFiltersAttachment}' attachment.");
        }

        var tokenizer = TokenizerData.FromMetadata(src.Metadata);
        var attachments = new Dictionary<string, IWeightSource>(src.Attachments, StringComparer.Ordinal)
        {
            [TokenizerAttachment] = new MemoryWeightSource(tokenizer.ToJson()),
        };
        var metadata = src.Metadata.Where(e => !e.Key.StartsWith("tokenizer.", StringComparison.Ordinal)).ToDictionary(StringComparer.Ordinal);
        int sampleRate = src.Arch<int>("audio.sample_rate");
        var featureConfig = new Dictionary<string, JsonElement>(StringComparer.Ordinal)
        {
            ["type"] = Json("log-mel"),
            ["sampleRate"] = Json(sampleRate),
            ["fftSize"] = Json(src.Arch<int>("audio.fft_size")),
            ["hopLength"] = Json(src.Arch<int>("audio.hop_length")),
            ["melBins"] = Json(src.Arch<int>("audio.mel_bins")),
            ["center"] = Json(true),
            ["padMode"] = Json("reflect"),
            ["trailingPadding"] = Json(src.Arch<int>("audio.chunk_samples")),
            ["dropLastFrame"] = Json(true),
            ["logBase"] = Json("10"),
            ["logFloor"] = Json(src.Arch("audio.log_floor", 1e-10)),
            ["dynamicRange"] = Json(src.Arch("audio.dynamic_range", 8.0)),
            ["logOffset"] = Json(src.Arch("audio.log_offset", 4.0)),
            ["logScale"] = Json(src.Arch("audio.log_scale", 0.25)),
        };
        var description = new ModelDescription
        {
            Kind = ModelKinds.SpeechRecognition,
            Components =
            [
                new ModelComponent { Role = ModelComponentRoles.Encoder, Graphs = [WellKnown.EncodeGraph], WeightPrefixes = ["encoder."] },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Decoder,
                    Graphs = [WellKnown.DecodeGraph],
                    WeightPrefixes = ["decoder.", "token_embedding.", "output."],
                    SymbolBounds = new Dictionary<string, long>(StringComparer.Ordinal)
                    {
                        [WellKnown.TokenSymbol] = decoderContext,
                        [WellKnown.ContextSymbol] = decoderContext,
                        [WellKnown.EncoderSymbol] = sourceRows,
                        [OutputRowsSymbol] = 1,
                    },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Attachments = new Dictionary<string, string>(StringComparer.Ordinal) { [ModelComponentAttachments.Tokenizer] = TokenizerAttachment },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.FeatureExtractor,
                    Config = featureConfig,
                    Attachments = new Dictionary<string, string>(StringComparer.Ordinal) { [ModelComponentAttachments.MelFilters] = MelFiltersAttachment },
                },
            ],
            Config = new Dictionary<string, JsonElement>(StringComparer.Ordinal) { ["sampleRate"] = Json(sampleRate) },
        };
        return (description, attachments, metadata);
    }

    private static JsonElement Json(string value) => Parse($"\"{JsonEncodedText.Encode(value)}\"");

    private static JsonElement Json(int value) => Parse(value.ToString(CultureInfo.InvariantCulture));

    private static JsonElement Json(double value) => Parse(value.ToString("R", CultureInfo.InvariantCulture));

    private static JsonElement Json(bool value) => Parse(value ? "true" : "false");

    private static JsonElement Parse(string json)
    {
        using var document = JsonDocument.Parse(json);
        return document.RootElement.Clone();
    }
}
