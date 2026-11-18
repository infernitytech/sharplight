using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The <see cref="ModelDescription"/> of an audio language model used as a speech recognizer (kind <see cref="ModelKinds.SpeechRecognition"/>):
/// the <c>encoder</c> component is the audio tower (<see cref="ChunkedAudioTowerGraph"/>, configured for the host-side chunk layout), the
/// <c>decoder</c> component the language model with its multimodal graph (configured with the <see cref="AudioPromptSettings"/>), the
/// <c>tokenizer</c> the language model's own (in the metadata) and the <c>feature-extractor</c> the log-mel front end of the source.
/// </summary>
internal static class AudioLanguageModelDescription
{
    /// <summary>The longest window the encoder takes, in seconds.</summary>
    public const int MaxWindowSeconds = 30;

    /// <summary>Builds the description.</summary>
    /// <param name="src">The model source (feature-extractor keys and the filter bank attachment).</param>
    /// <param name="tower">The audio tower.</param>
    /// <param name="prompt">The prompt settings.</param>
    /// <param name="graphs">All graphs of the model; every graph but the tower's belongs to the decoder.</param>
    /// <param name="weights">All weights; every weight outside <see cref="AudioTowerParameters.WeightPrefix"/> belongs to the decoder.</param>
    public static ModelDescription Build(ModelSource src, AudioTowerParameters tower, AudioPromptSettings prompt, IReadOnlyDictionary<string, ComputeGraph> graphs,
        IReadOnlyDictionary<string, WeightInfo> weights)
    {
        int sampleRate = src.Arch<int>("audio.sample_rate");
        int hop = src.Arch<int>("audio.hop_length");
        bool causal = tower.Tower == AudioTowerParameters.Causal;
        bool whole = tower.Tower == AudioTowerParameters.Whisper;
        int maxFrames = whole || causal ? tower.WindowFrames : MaxWindowSeconds * sampleRate / hop;
        var layout = new Speech.ChunkedAudioLayout(tower.ChunkFrames, tower.WindowFrames, tower.ConvCount, tower.MelBins);
        var decoderPrefixes = weights.Keys
            .Where(k => !k.StartsWith(AudioTowerParameters.WeightPrefix, StringComparison.Ordinal))
            .Select(k => k.IndexOf('.', StringComparison.Ordinal) is var dot and > 0 ? k[..(dot + 1)] : k)
            .Distinct()
            .Order(StringComparer.Ordinal)
            .ToList();
        var encoderConfig = new Dictionary<string, object>(tower.ComponentConfig(), StringComparer.Ordinal)
        {
            ["maxFrames"] = maxFrames,
            ["minFrames"] = src.Arch("audio.min_samples", 0) / hop,
        };
        return new ModelDescription
        {
            Kind = ModelKinds.SpeechRecognition,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.Encoder,
                    Graphs = [ChunkedAudioTowerGraph.AudioEncodeGraph],
                    WeightPrefixes = [AudioTowerParameters.WeightPrefix],
                    Config = SpeechRecognitionDescription.JsonConfig(encoderConfig),
                    SymbolBounds = causal
                        ? new Dictionary<string, long>(StringComparer.Ordinal)
                        {
                            [WellKnown.AudioFrameSymbol] = maxFrames,
                            [WellKnown.EncoderSymbol] = maxFrames / (2 * tower.Downsample),
                        }
                        : whole
                        ? new Dictionary<string, long>(StringComparer.Ordinal)
                        : new Dictionary<string, long>(StringComparer.Ordinal)
                        {
                            [WellKnown.AudioFrameSymbol] = (long)layout.Chunks(maxFrames) * layout.PaddedFrames,
                            [WellKnown.EncoderSymbol] = layout.OutputRows(maxFrames),
                        },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Decoder,
                    Graphs = [.. graphs.Keys.Where(g => g != ChunkedAudioTowerGraph.AudioEncodeGraph).Order(StringComparer.Ordinal)],
                    WeightPrefixes = decoderPrefixes,
                    Config = SpeechRecognitionDescription.JsonConfig(prompt.ComponentConfig()),
                },
                new ModelComponent { Role = ModelComponentRoles.Tokenizer, MetadataPrefix = null },
                SpeechRecognitionDescription.FeatureExtractor(src, 0),
            ],
            Config = ComponentConfig.Of(("sampleRate", sampleRate)),
        };
    }
}
