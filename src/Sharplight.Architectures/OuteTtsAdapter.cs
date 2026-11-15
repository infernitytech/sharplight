using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// OuteTTS: a decoder language model that generates codec tokens after a text prompt, plus the WavTokenizer decoder that turns the tokens into
/// a waveform. A composite adapter (<see cref="ModelAdapterRegistry.RegisterComposite"/>): the primary source is the language model, built by
/// whichever adapter handles its architecture (OuteTTS 0.2 is a Qwen2 decoder), the <see cref="AuxiliaryRoles.Vocoder"/> source the codec
/// (<see cref="WavTokenizerGraph"/>, weights under <c>codec.</c>). The result is a <see cref="ModelKinds.SpeechSynthesis"/> model with the
/// components <c>text-model</c> (the decoder graphs), <c>tokenizer</c> and <c>codec-decoder</c>; <see cref="OuteTtsSynthesizer"/> runs it.
/// </summary>
internal sealed class OuteTtsAdapter : IModelAdapter
{
    /// <summary>The role of the language model component.</summary>
    public const string TextModelRole = "text-model";

    /// <summary>The role of the codec decoder component.</summary>
    public const string CodecRole = "codec-decoder";

    /// <summary>The prefix of the codec's weights in the model.</summary>
    public const string CodecPrefix = "codec.";

    /// <summary>The largest number of codec frames one decode runs over (about 55 s of speech at 75 frames per second).</summary>
    public const int MaxCodecFrames = 4096;

    /// <summary>The model family recorded in the language model component's configuration.</summary>
    public const string Family = "outetts";

    /// <inheritdoc/>
    public string Name => "OuteTTS";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures { get; } = [];

    /// <inheritdoc/>
    public bool CanLoad(ModelSourceSet sources)
        => sources.Auxiliary.TryGetValue(AuxiliaryRoles.Vocoder, out var codec) && codec.Architecture == WavTokenizerParameters.Architecture
           && OuteTtsPrompt.VersionOf(sources.Primary) is not null;

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var primary = sources.Primary;
        string version = OuteTtsPrompt.VersionOf(primary) ?? throw new NotSupportedException("The language model is not an OuteTTS checkpoint.");
        var codecSource = sources.Auxiliary[AuxiliaryRoles.Vocoder];
        var textAdapter = ModelAdapterRegistry.Find(primary) ?? throw new NotSupportedException($"No adapter for the OuteTTS language model architecture '{primary.Architecture}'.");
        var text = textAdapter.Build(new ModelSourceSet { Primary = primary }, options);

        var codec = WavTokenizerParameters.From(codecSource);
        var weights = new Dictionary<string, WeightInfo>(text.Weights, StringComparer.Ordinal);
        foreach (var (name, weight) in codecSource.Weights)
        {
            weights[CodecPrefix + name] = weight.As(CodecPrefix + name);
        }

        var graphs = new Dictionary<string, Graph.ComputeGraph>(text.Graphs, StringComparer.Ordinal)
        {
            [WavTokenizerGraph.GraphName] = WavTokenizerGraph.Build(codec, weights.GetValueOrDefault, CodecPrefix),
        };
        var textPrefixes = text.Weights.Keys.Select(k => k[..(k.IndexOf('.', StringComparison.Ordinal) + 1)]).Where(p => p.Length > 0).Distinct().Order(StringComparer.Ordinal).ToList();
        var description = new ModelDescription
        {
            Kind = ModelKinds.SpeechSynthesis,
            Components =
            [
                new ModelComponent
                {
                    Role = TextModelRole,
                    Graphs = [.. text.Graphs.Keys.Order(StringComparer.Ordinal)],
                    WeightPrefixes = textPrefixes,
                    Config = ComponentConfig.Of(("family", Family), ("version", version)),
                },
                new ModelComponent { Role = ModelComponentRoles.Tokenizer, MetadataPrefix = null },
                new ModelComponent
                {
                    Role = CodecRole,
                    Graphs = [WavTokenizerGraph.GraphName],
                    WeightPrefixes = [CodecPrefix],
                    Config = ComponentConfig.Of(("architecture", WavTokenizerParameters.Architecture), ("sampleRate", codec.SampleRate), ("frameRate", codec.SampleRate / codec.HopLength),
                        ("nFft", codec.NFft), ("hopLength", codec.HopLength), ("vocabSize", codec.VocabSize), ("window", "hann")),
                    SymbolBounds = new Dictionary<string, long> { [WavTokenizerGraph.FrameSymbol] = MaxCodecFrames },
                },
            ],
            Config = ComponentConfig.Of(("sampleRate", codec.SampleRate)),
        };
        return new ModelDefinition
        {
            Architecture = text.Architecture,
            Name = text.Name,
            Weights = weights,
            Graphs = graphs,
            States = text.States,
            Properties = text.Properties,
            SourceMetadata = text.SourceMetadata,
            Description = description,
            Attachments = text.Attachments,
            OwnedResources = [.. text.OwnedResources, .. codecSource.OwnedResources],
        };
    }
}
