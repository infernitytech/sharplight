using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// F5-TTS (canonical architecture <c>f5tts</c>) with its Vocos vocoder: a composite adapter (<see cref="ICompositeModelAdapter"/>) whose primary source is
/// the flow-matching transformer and whose <see cref="AuxiliaryRoles.Vocoder"/> source is Vocos (weights under <c>vocoder.</c>). The model is a
/// <see cref="ModelKinds.SpeechSynthesis"/> model with the components <c>transformer</c> (<see cref="F5TransformerGraph"/>), <c>vocoder</c>
/// (<see cref="VocosGraph"/>), <c>tokenizer</c> (the character vocabulary attachment) and <c>feature-extractor</c> (the reference audio's log-mel);
/// <see cref="F5Synthesizer"/> runs it. It speaks with the voice of a reference recording given with its text.
/// </summary>
internal sealed class F5Adapter : ICompositeModelAdapter
{
    /// <summary>The role of the transformer component.</summary>
    public const string TransformerRole = "transformer";

    /// <summary>The role of the vocoder component.</summary>
    public const string VocoderRole = "vocoder";

    /// <summary>The prefix of the vocoder's weights.</summary>
    public const string VocoderPrefix = "vocoder.";

    /// <summary>The most mel frames one request covers (about 87 s at 24 kHz with a hop of 256).</summary>
    public const int MaxFrames = 8192;

    /// <inheritdoc/>
    public string Name => "F5-TTS";

    /// <inheritdoc/>
    public bool CanLoad(ModelSourceSet sources)
        => sources.Primary.Architecture == F5Parameters.Architecture
           && sources.Auxiliary.TryGetValue(AuxiliaryRoles.Vocoder, out var vocoder) && vocoder.Architecture == "vocos";

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options, ArchitectureCatalog architectures)
    {
        var src = sources.Primary;
        var vocos = sources.Auxiliary[AuxiliaryRoles.Vocoder];
        var p = F5Parameters.From(src);
        var weights = new Dictionary<string, WeightInfo>(src.Weights, StringComparer.Ordinal);
        foreach (var (name, weight) in vocos.Weights)
        {
            weights[VocoderPrefix + name] = weight.As(VocoderPrefix + name);
        }

        int vocosMels = vocos.Get<int>("vocos.mel_bins");
        if (vocosMels != p.MelBins)
        {
            throw new NotSupportedException($"The vocoder takes {vocosMels} mel bins, the F5-TTS model makes {p.MelBins}.");
        }

        int nFft = vocos.Get<int>("vocos.fft_size");
        int hop = vocos.Get<int>("vocos.hop_length");
        var graphs = new Dictionary<string, ComputeGraph>(StringComparer.Ordinal)
        {
            [F5TransformerGraph.GraphName] = F5TransformerGraph.Build(p, weights.GetValueOrDefault),
            [VocosGraph.GraphName] = VocosGraph.Build(weights.GetValueOrDefault, VocoderPrefix, p.MelBins, vocos.Get<int>("vocos.layers"), nFft),
        };
        int sampleRate = src.Arch<int>("audio.sample_rate");
        var features = ComponentConfig.Of(("type", "log-mel"), ("sampleRate", sampleRate), ("fftSize", src.Arch<int>("audio.fft_size")),
            ("hopLength", src.Arch<int>("audio.hop_length")), ("melBins", p.MelBins), ("center", true), ("padMode", "reflect"), ("magnitude", true),
            ("logBase", "e"), ("logFloor", 1e-5));
        var description = new ModelDescription
        {
            Kind = ModelKinds.SpeechSynthesis,
            Components =
            [
                new ModelComponent
                {
                    Role = TransformerRole,
                    Graphs = [F5TransformerGraph.GraphName],
                    WeightPrefixes = ["f5."],
                    Config = ComponentConfig.Of(("steps", p.Steps), ("guidance", p.Guidance), ("sway", p.Sway), ("melBins", p.MelBins)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxFrames },
                },
                new ModelComponent
                {
                    Role = VocoderRole,
                    Graphs = [VocosGraph.GraphName],
                    WeightPrefixes = [VocoderPrefix],
                    Config = ComponentConfig.Of(("fftSize", nFft), ("hop", hop)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxFrames },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Attachments = new Dictionary<string, string>(StringComparer.Ordinal) { ["characters"] = "tokenizer/characters.json" },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.FeatureExtractor,
                    Config = features,
                    Attachments = new Dictionary<string, string>(StringComparer.Ordinal) { [ModelComponentAttachments.MelFilters] = "feature-extractor/mel-filters" },
                },
            ],
            Config = ComponentConfig.Of(("sampleRate", sampleRate)),
        };
        return new ModelDefinition
        {
            Architecture = F5Parameters.Architecture,
            Name = src.Get(CanonicalKeys.Name, F5Parameters.Architecture),
            Weights = weights,
            Graphs = graphs,
            SourceMetadata = src.Metadata,
            Description = description,
            Attachments = src.Attachments,
            OwnedResources = [.. src.OwnedResources, .. vocos.OwnedResources],
        };
    }
}
