using System.Globalization;
using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the <see cref="ModelDescription"/> of an encoder-decoder speech recognizer (kind <see cref="ModelKinds.SpeechRecognition"/>):
/// <c>encoder</c> and <c>decoder</c> components over the model's graphs, a <c>tokenizer</c> component whose tokenizer is stored as an
/// attachment (<see cref="ModelComponentAttachments.Tokenizer"/>, the canonical <see cref="TokenizerData"/> JSON) and a
/// <c>feature-extractor</c> component configured from the canonical <c>{arch}.audio.*</c> hyper-parameters (log-mel with its filter bank
/// attachment, or the raw waveform). The tokenizer keys leave the definition's metadata, so the package stores the vocabulary once.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>{arch}.audio.</c>): <c>features</c> ("log-mel", the default, or "waveform"), <c>sample_rate</c>. Log-mel:
/// <c>fft_size</c>, <c>hop_length</c>, <c>mel_bins</c>, <c>chunk_samples</c> (the encoder window, also appended as silence so every window is
/// complete), optional <c>log_floor</c> (default 1e-10), <c>dynamic_range</c> (log10 units below the recording's peak, default 8),
/// <c>log_offset</c> (default 4) and <c>log_scale</c> (default 0.25): Whisper's normalization; optional <c>window_size</c>,
/// <c>periodic_window</c>, <c>preemphasis</c>, <c>pad_mode</c>, <c>log_base</c>, <c>log_guard</c> and <c>mel_normalize</c> (NeMo's front end: a
/// symmetric 400-sample window, pre-emphasis 0.97, zero padding, ln(mel + 2⁻²⁴), per-channel normalization; see <c>LogMelSpectrogram</c>); source attachment
/// <see cref="MelFiltersAttachment"/>: f32 [mel_bins, fft_size / 2 + 1]. Waveform: optional <c>normalize</c> (zero mean and unit variance per
/// recording, default false). Variable-length encoders (<c>audio.variable_length</c>) take up to <c>audio.max_frames</c> frames per window
/// (the encoder component's bound of <see cref="WellKnown.AudioFrameSymbol"/>) and append no padding.
/// </remarks>
internal static class SpeechRecognitionDescription
{
    /// <summary>The canonical source attachment of the mel filter bank.</summary>
    public const string MelFiltersAttachment = "feature-extractor/mel-filters";

    /// <summary>The attachment that stores the tokenizer.</summary>
    public const string TokenizerAttachment = "tokenizer/tokenizer.json";

    /// <summary>The symbol of the decoder's logits rows (the speech decoder reads the last row only).</summary>
    private const string OutputRowsSymbol = "R";

    /// <summary>
    /// The description of a prompted encoder-decoder recognizer (Whisper, Moonshine, Canary), the definition's attachments and its metadata
    /// without the tokenizer keys.
    /// </summary>
    /// <param name="src">The model source (hyper-parameters, tokenizer metadata, filter bank attachment).</param>
    /// <param name="decoderContext">The decoder's context length in tokens.</param>
    /// <param name="sourceRows">Encoder output rows per window (the most rows of a variable-length encoder).</param>
    /// <param name="maxFrames">The most frames of a variable-length encoder's window, or 0 for a fixed window.</param>
    /// <exception cref="NotSupportedException">The source has no filter bank or no tokenizer.</exception>
    public static (ModelDescription Description, IReadOnlyDictionary<string, IWeightSource> Attachments, IReadOnlyDictionary<string, object> Metadata) Build(
        ModelSource src, int decoderContext, int sourceRows, long maxFrames)
    {
        var encoderBounds = new Dictionary<string, long>(StringComparer.Ordinal);
        if (maxFrames > 0)
        {
            encoderBounds[WellKnown.AudioFrameSymbol] = maxFrames;
        }

        var encoder = new ModelComponent { Role = ModelComponentRoles.Encoder, Graphs = [WellKnown.EncodeGraph], WeightPrefixes = ["encoder."], SymbolBounds = encoderBounds };
        var decoder = new ModelComponent
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
        };
        return Build(src, encoder, decoder, maxFrames);
    }

    /// <summary>
    /// The description of a speech recognizer with the given encoder and decoder components plus the tokenizer and feature-extractor
    /// components, the definition's attachments and its metadata without the tokenizer keys.
    /// </summary>
    /// <param name="src">The model source (hyper-parameters, tokenizer metadata, filter bank attachment).</param>
    /// <param name="encoder">The encoder component.</param>
    /// <param name="decoder">The decoder component (graphs, or only the configuration of a host-side decoding such as CTC).</param>
    /// <param name="maxFrames">The most frames of a variable-length encoder's window, or 0 for a fixed window.</param>
    /// <exception cref="NotSupportedException">The source has no filter bank or no tokenizer.</exception>
    public static (ModelDescription Description, IReadOnlyDictionary<string, IWeightSource> Attachments, IReadOnlyDictionary<string, object> Metadata) Build(
        ModelSource src, ModelComponent encoder, ModelComponent decoder, long maxFrames)
    {
        var featureComponent = FeatureExtractor(src, maxFrames);
        var tokenizer = TokenizerData.FromMetadata(src.Metadata);
        var attachments = new Dictionary<string, IWeightSource>(src.Attachments, StringComparer.Ordinal)
        {
            [TokenizerAttachment] = new MemoryWeightSource(tokenizer.ToJson()),
        };
        var metadata = src.Metadata.Where(e => !e.Key.StartsWith("tokenizer.", StringComparison.Ordinal)).ToDictionary(StringComparer.Ordinal);
        int sampleRate = src.Arch<int>("audio.sample_rate");
        var description = new ModelDescription
        {
            Kind = ModelKinds.SpeechRecognition,
            Components =
            [
                encoder,
                decoder,
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Attachments = new Dictionary<string, string>(StringComparer.Ordinal) { [ModelComponentAttachments.Tokenizer] = TokenizerAttachment },
                },
                featureComponent,
            ],
            Config = new Dictionary<string, JsonElement>(StringComparer.Ordinal) { ["sampleRate"] = Json(sampleRate) },
        };
        return (description, attachments, metadata);
    }

    /// <summary>A component configuration of strings, numbers, booleans and number arrays as JSON values.</summary>
    /// <param name="values">The values by key.</param>
    /// <exception cref="NotSupportedException">A value has another type.</exception>
    public static IReadOnlyDictionary<string, JsonElement> JsonConfig(IReadOnlyDictionary<string, object> values)
        => values.ToDictionary(e => e.Key, e => e.Value switch
        {
            string text => Json(text),
            bool flag => Json(flag),
            int number => Json(number),
            long number => Parse(number.ToString(CultureInfo.InvariantCulture)),
            double number => Json(number),
            float number => Json((double)number),
            long[] numbers => Parse("[" + string.Join(",", numbers.Select(n => n.ToString(CultureInfo.InvariantCulture))) + "]"),
            int[] numbers => Parse("[" + string.Join(",", numbers.Select(n => n.ToString(CultureInfo.InvariantCulture))) + "]"),
            var other => throw new NotSupportedException($"Component configuration value of type {other.GetType().Name}."),
        }, StringComparer.Ordinal);

    /// <summary>
    /// The feature-extractor component of the source's <c>audio.features</c> (the attachments it names must be in the source).
    /// </summary>
    /// <param name="src">The model source.</param>
    /// <param name="maxFrames">The most frames of a variable-length window, or 0 for a fixed window (whose length is padded to).</param>
    /// <exception cref="NotSupportedException">The feature type is unknown or its attachment is missing.</exception>
    public static ModelComponent FeatureExtractor(ModelSource src, long maxFrames)
    {
        int sampleRate = src.Arch<int>("audio.sample_rate");
        string type = src.Arch("audio.features", "log-mel");
        if (type == "waveform")
        {
            var waveform = new Dictionary<string, JsonElement>(StringComparer.Ordinal)
            {
                ["type"] = Json("waveform"),
                ["sampleRate"] = Json(sampleRate),
                ["normalize"] = Json(src.Arch("audio.normalize", false)),
            };
            return new ModelComponent { Role = ModelComponentRoles.FeatureExtractor, Config = waveform };
        }

        if (type != "log-mel")
        {
            throw new NotSupportedException($"{src.Architecture}: unknown audio.features '{type}'.");
        }

        if (!src.Attachments.ContainsKey(MelFiltersAttachment))
        {
            throw new NotSupportedException($"{src.Architecture}: a log-mel speech encoder needs the '{MelFiltersAttachment}' attachment.");
        }

        var featureConfig = new Dictionary<string, JsonElement>(StringComparer.Ordinal)
        {
            ["type"] = Json("log-mel"),
            ["sampleRate"] = Json(sampleRate),
            ["fftSize"] = Json(src.Arch<int>("audio.fft_size")),
            ["hopLength"] = Json(src.Arch<int>("audio.hop_length")),
            ["melBins"] = Json(src.Arch<int>("audio.mel_bins")),
            ["windowSize"] = Json(src.Arch("audio.window_size", src.Arch<int>("audio.fft_size"))),
            ["periodicWindow"] = Json(src.Arch("audio.periodic_window", true)),
            ["preemphasis"] = Json(src.Arch("audio.preemphasis", 0.0)),
            ["center"] = Json(true),
            ["padMode"] = Json(src.Arch("audio.pad_mode", "reflect")),
            ["trailingPadding"] = Json(maxFrames > 0 ? 0 : src.Arch<int>("audio.chunk_samples")),
            ["dropLastFrame"] = Json(true),
            ["logBase"] = Json(src.Arch("audio.log_base", "10")),
            ["logFloor"] = Json(src.Arch("audio.log_floor", 1e-10)),
            ["logGuard"] = Json(src.Arch("audio.log_guard", 0.0)),
            ["dynamicRange"] = Json(src.Arch("audio.dynamic_range", 8.0)),
            ["logOffset"] = Json(src.Arch("audio.log_offset", 4.0)),
            ["logScale"] = Json(src.Arch("audio.log_scale", 0.25)),
            ["normalize"] = Json(src.Arch("audio.mel_normalize", "none")),
        };
        return new ModelComponent
        {
            Role = ModelComponentRoles.FeatureExtractor,
            Config = featureConfig,
            Attachments = new Dictionary<string, string>(StringComparer.Ordinal) { [ModelComponentAttachments.MelFilters] = MelFiltersAttachment },
        };
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
