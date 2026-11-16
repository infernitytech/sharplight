using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Orpheus (a Llama decoder that generates SNAC codec tokens after a text prompt) with its SNAC decoder: a composite adapter
/// (<see cref="ICompositeModelAdapter"/>) whose primary source is the language model, built by whichever adapter handles its architecture, and whose
/// <see cref="AuxiliaryRoles.Vocoder"/> source is the SNAC codec (<see cref="SnacGraph"/>, weights under <c>codec.</c>). The result is a
/// <see cref="ModelKinds.SpeechSynthesis"/> model with the components <c>text-model</c>, <c>tokenizer</c> and <c>codec-decoder</c>;
/// <see cref="OrpheusSynthesizer"/> runs it.
/// </summary>
internal sealed class OrpheusAdapter : ICompositeModelAdapter
{
    /// <summary>The role of the language model component.</summary>
    public const string TextModelRole = "text-model";

    /// <summary>The role of the codec decoder component.</summary>
    public const string CodecRole = "codec-decoder";

    /// <summary>The prefix of the codec's weights in the model.</summary>
    public const string CodecPrefix = "codec.";

    /// <summary>The model family recorded in the language model component's configuration.</summary>
    public const string Family = "orpheus";

    /// <summary>The largest number of coarse codec frames one decode runs over (about 85 s of speech at 12 frames per second).</summary>
    public const int MaxCodecFrames = 1024;

    /// <summary>The voices of the Orpheus fine-tune (the prompt names one: <c>"tara: Hello"</c>).</summary>
    public static readonly string[] DefaultVoices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"];

    /// <inheritdoc/>
    public string Name => "Orpheus";

    /// <inheritdoc/>
    public bool CanLoad(ModelSourceSet sources)
        => sources.Auxiliary.TryGetValue(AuxiliaryRoles.Vocoder, out var codec) && codec.Architecture == SnacParameters.Architecture;

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options, ArchitectureCatalog architectures)
    {
        var primary = sources.Primary;
        var codecSource = sources.Auxiliary[AuxiliaryRoles.Vocoder];
        if (!architectures.TryFindAdapter(primary, out var textAdapter))
        {
            throw new NotSupportedException($"No adapter for the Orpheus language model architecture '{primary.Architecture}'.");
        }

        var text = textAdapter.Build(new ModelSourceSet { Primary = primary }, options);
        var codec = SnacParameters.From(codecSource);
        var weights = new Dictionary<string, WeightInfo>(text.Weights, StringComparer.Ordinal);
        foreach (var (name, weight) in codecSource.Weights)
        {
            weights[CodecPrefix + name] = weight.As(CodecPrefix + name);
        }

        var graphs = new Dictionary<string, ComputeGraph>(text.Graphs, StringComparer.Ordinal)
        {
            [SnacGraph.GraphName] = SnacGraph.Build(codec, weights.GetValueOrDefault, CodecPrefix),
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
                    Config = ComponentConfig.Of(("family", Family)),
                },
                new ModelComponent { Role = ModelComponentRoles.Tokenizer, MetadataPrefix = null },
                new ModelComponent
                {
                    Role = CodecRole,
                    Graphs = [SnacGraph.GraphName],
                    WeightPrefixes = [CodecPrefix],
                    Config = ComponentConfig.Of(("architecture", SnacParameters.Architecture), ("sampleRate", codec.SampleRate), ("codebooks", codec.VqStrides.Length),
                        ("codebookSize", codec.CodebookSize), ("samplesPerFrame", codec.SamplesPerCoarseCode), ("noise", codec.Noise),
                        ("strides", string.Join(',', codec.VqStrides)), ("rates", string.Join(',', codec.DecoderRates))),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxCodecFrames },
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
