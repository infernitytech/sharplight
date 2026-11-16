using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Kokoro (StyleTTS 2 style text-to-speech, canonical architecture <c>kokoro</c>, see <c>Sharplight.Import.Kokoro</c>): builds the two graphs
/// (<see cref="KokoroTextGraph"/> over the phoneme symbols, <see cref="KokoroAcousticGraph"/> over the decoder frames) and describes the model as
/// <see cref="ModelKinds.SpeechSynthesis"/> with the components <c>text-encoder</c> and <c>acoustic-decoder</c>; <see cref="KokoroSynthesizer"/> runs it.
/// The voice packs and the symbol vocabulary are attachments of the text-encoder component.
/// </summary>
internal sealed class KokoroAdapter : IModelAdapter
{
    /// <summary>The role of the component running <see cref="KokoroTextGraph"/>.</summary>
    public const string TextRole = "text-encoder";

    /// <summary>The role of the component running <see cref="KokoroAcousticGraph"/>.</summary>
    public const string AcousticRole = "acoustic-decoder";

    /// <summary>The attachment key of the symbol vocabulary (JSON: symbol to id).</summary>
    public const string VocabularyKey = "vocabulary";

    /// <summary>The prefix of the attachment keys of the voices.</summary>
    public const string VoicePrefix = "voices/";

    /// <summary>The prefix of the attachment keys of the pronunciation lexicons (<c>lexicon/us_gold.json</c>, ...).</summary>
    public const string LexiconPrefix = "lexicon/";

    /// <summary>The suffix of the attachment keys of the voices.</summary>
    public const string VoiceSuffix = ".f32";

    /// <summary>The model family recorded in the configuration of the text component.</summary>
    public const string Family = "kokoro";

    /// <summary>The largest number of decoder frames one acoustic run covers (about 75 s of speech).</summary>
    public const int MaxFrames = 3000;

    /// <inheritdoc/>
    public string Name => "Kokoro";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [KokoroParameters.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = KokoroParameters.From(src);
        var graphs = new Dictionary<string, ComputeGraph>(StringComparer.Ordinal)
        {
            [KokoroTextGraph.GraphName] = KokoroTextGraph.Build(p, src.Weight),
            [KokoroAcousticGraph.GraphName] = KokoroAcousticGraph.Build(p, src.Weight),
        };
        var attachments = new Dictionary<string, string>(StringComparer.Ordinal) { [VocabularyKey] = "tokenizer/vocab.json" };
        foreach (string key in src.Attachments.Keys.Where(k => k.StartsWith(VoicePrefix, StringComparison.Ordinal) || k.StartsWith(LexiconPrefix, StringComparison.Ordinal))
                     .Order(StringComparer.Ordinal))
        {
            attachments[key] = key;
        }

        var description = new ModelDescription
        {
            Kind = ModelKinds.SpeechSynthesis,
            Components =
            [
                new ModelComponent
                {
                    Role = TextRole,
                    Graphs = [KokoroTextGraph.GraphName],
                    WeightPrefixes = ["bert.", "bert_encoder.", "text_encoder.", "predictor.text_encoder.", "predictor.lstm.", "predictor.duration_proj."],
                    Config = ComponentConfig.Of(("family", Family), ("contextLength", p.BertContext), ("styleDim", p.StyleDim)),
                    Attachments = attachments,
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = p.BertContext },
                },
                new ModelComponent
                {
                    Role = AcousticRole,
                    Graphs = [KokoroAcousticGraph.GraphName],
                    WeightPrefixes = ["predictor.shared.", "predictor.F0", "predictor.N", "decoder."],
                    Config = ComponentConfig.Of(("sampleRate", p.SampleRate), ("samplesPerFrame", p.SamplesPerFrame), ("nFft", p.NFft), ("hop", p.Hop),
                        ("harmonics", p.Harmonics), ("styleDim", p.StyleDim)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxFrames },
                },
            ],
            Config = ComponentConfig.Of(("sampleRate", p.SampleRate)),
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = graphs,
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
            Attachments = src.Attachments,
        };
    }
}
