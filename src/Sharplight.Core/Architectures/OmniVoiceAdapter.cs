using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// OmniVoice (canonical architecture <see cref="Architecture"/>, k2-fsa's massively multilingual text-to-speech model) with its audio tokenizer: a composite adapter
/// (<see cref="ICompositeModelAdapter"/>) whose primary source is the language model and whose <see cref="AuxiliaryRoles.Vocoder"/> source is the Higgs Audio v2 tokenizer
/// (<see cref="HiggsAudioCodecAdapter"/>'s architecture, weights under <c>codec.</c>). The language model is a Qwen3 backbone with bidirectional attention over text tokens and eight
/// codebooks of audio tokens (a position's embedding is the sum of its codebooks' embeddings), built by <see cref="DecoderAdapter"/> as a masked-diffusion model whose
/// <see cref="WellKnown.DenoiseGraph"/> takes embedding rows and returns the eight audio heads' logits per row; the tokenizer's encoder (reference audio to codes) and decoder (codes to
/// 24 kHz audio) are the codec graphs. <see cref="OmniVoiceSynthesizer"/> runs the iterative unmasking.
/// </summary>
/// <remarks>
/// Hyper-parameters: the backbone's decoder keys (<c>decoder.*</c>, <c>decoder.attention.causal</c> false, <c>decoder.embedding_input</c> true) and <c>omnivoice.{codebooks,audio_vocab,mask_id,
/// codebook_weights}</c>. Weights: the canonical decoder names (<c>token_embedding.weight</c> is the text embedding, <c>lm_head.weight</c> the eight audio heads [codebooks · audio_vocab, hidden]),
/// <c>omnivoice.audio_embedding.weight</c> [codebooks · audio_vocab, hidden] and the codec under <c>codec.</c>.
/// </remarks>
internal sealed class OmniVoiceAdapter : ICompositeModelAdapter
{
    /// <summary>The canonical architecture id of the language model.</summary>
    public const string Architecture = "omnivoice";

    /// <summary>The role of the language model component.</summary>
    public const string TextModelRole = "text-model";

    /// <summary>The role of the codec decoder component.</summary>
    public const string CodecRole = "codec-decoder";

    /// <summary>The role of the codec encoder component.</summary>
    public const string CodecEncoderRole = "codec-encoder";

    /// <summary>The prefix of the codec's weights in the model.</summary>
    public const string CodecPrefix = "codec.";

    /// <summary>The model family recorded in the language model component's configuration.</summary>
    public const string Family = "omnivoice";

    /// <summary>The most rows (the conditional and the unconditional sequence) one pass handles.</summary>
    public const int MaxRows = 4096;

    /// <summary>The most audio frames one pass generates (30 s at 25 frames per second).</summary>
    public const int MaxTargetFrames = 750;

    /// <inheritdoc/>
    public string Name => "OmniVoice";

    /// <inheritdoc/>
    public bool CanLoad(ModelSourceSet sources)
        => sources.Primary.Architecture == Architecture && sources.Auxiliary.TryGetValue(AuxiliaryRoles.Vocoder, out var codec)
           && codec.Architecture == HiggsAudioCodecParameters.Architecture;

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options, ArchitectureCatalog architectures)
    {
        var src = sources.Primary;
        var codecSource = sources.Auxiliary[AuxiliaryRoles.Vocoder];
        string a = Architecture + ".";
        var backboneSource = new ModelSource
        {
            Format = src.Format,
            Architecture = "decoder",
            Metadata = new Dictionary<string, object>(src.Metadata, StringComparer.Ordinal) { ["decoder.embedding_input"] = true },
            Weights = src.Weights,
            Attachments = src.Attachments,
            Path = src.Path,
        };
        var backbone = new DecoderAdapter().Build(new ModelSourceSet { Primary = backboneSource }, options);
        if (!backbone.Graphs.ContainsKey(WellKnown.DenoiseGraph))
        {
            throw new NotSupportedException($"{Architecture}: the backbone has no denoise graph (it must be bidirectional with diffusion.shift_logits declared).");
        }

        var codec = HiggsAudioCodecParameters.From(codecSource);
        var kernel = SincResampleKernel.Create(codec.SampleRate, codec.SemanticSampleRate);
        var weights = new Dictionary<string, WeightInfo>(backbone.Weights, StringComparer.Ordinal);
        foreach (var (name, weight) in codecSource.Weights)
        {
            weights[CodecPrefix + name] = weight.As(CodecPrefix + name);
        }

        weights[CodecPrefix + HiggsAudioCodecGraph.ResampleKernel] = kernel.ToWeight(CodecPrefix + HiggsAudioCodecGraph.ResampleKernel);
        WeightInfo? Lookup(string n) => weights.GetValueOrDefault(CodecPrefix + n);
        var graphs = new Dictionary<string, ComputeGraph>(backbone.Graphs, StringComparer.Ordinal)
        {
            [AudioCodecGraphs.EncodeGraph] = HiggsAudioCodecGraph.BuildEncoder(codec, kernel, Lookup),
            [AudioCodecGraphs.DecodeGraph] = HiggsAudioCodecGraph.BuildDecoder(codec, Lookup),
        };
        var config = new List<(string, object)>
        {
            ("family", Family), ("codebooks", src.Get<int>(a + "codebooks")), ("audioVocab", src.Get<int>(a + "audio_vocab")), ("maskId", src.Get<int>(a + "mask_id")),
            ("codebookWeights", string.Join(',', src.Get<float[]>(a + "codebook_weights").Select(w => w.ToString(System.Globalization.CultureInfo.InvariantCulture)))),
            ("frameRate", (int)Math.Ceiling((double)codec.SampleRate / codec.SamplesPerFrame)),
        };
        var backbonePrefixes = backbone.Weights.Keys.Select(k => k[..(k.IndexOf('.', StringComparison.Ordinal) + 1)]).Where(p => p.Length > 0).Distinct().Order(StringComparer.Ordinal).ToList();
        var codecConfig = ComponentConfig.Of((AudioCodecGraphs.SampleRateKey, codec.SampleRate), (AudioCodecGraphs.SamplesPerFrameKey, (int)codec.SamplesPerFrame),
            (AudioCodecGraphs.CodebooksKey, codec.Codebooks), (AudioCodecGraphs.DefaultCodebooksKey, codec.Codebooks), (AudioCodecGraphs.CodebookSizeKey, codec.CodebookSize));
        var description = new ModelDescription
        {
            Kind = ModelKinds.SpeechSynthesis,
            Components =
            [
                new ModelComponent
                {
                    Role = TextModelRole,
                    Graphs = [WellKnown.DenoiseGraph],
                    WeightPrefixes = [.. backbonePrefixes, a + "audio_embedding."],
                    Config = ComponentConfig.Of([.. config]),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxRows },
                },
                new ModelComponent { Role = ModelComponentRoles.Tokenizer, MetadataPrefix = null },
                new ModelComponent
                {
                    Role = CodecEncoderRole,
                    Graphs = [AudioCodecGraphs.EncodeGraph],
                    WeightPrefixes = [CodecPrefix],
                    Config = codecConfig,
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = HiggsAudioCodecAdapter.MaxFrames * codec.SamplesPerFrame },
                },
                new ModelComponent
                {
                    Role = CodecRole,
                    Graphs = [AudioCodecGraphs.DecodeGraph],
                    WeightPrefixes = [CodecPrefix],
                    Config = codecConfig,
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = HiggsAudioCodecAdapter.MaxFrames },
                },
            ],
            Config = ComponentConfig.Of(("sampleRate", codec.SampleRate)),
        };
        var properties = new Dictionary<string, object>(backbone.Properties) { [WellKnown.ContextLength] = MaxRows };
        return new ModelDefinition
        {
            Architecture = Architecture,
            Name = backbone.Name,
            Weights = weights,
            Graphs = graphs,
            States = [.. backbone.States],
            Properties = properties,
            SourceMetadata = backbone.SourceMetadata,
            Description = description,
            Attachments = backbone.Attachments,
            OwnedResources = [.. backbone.OwnedResources, .. codecSource.OwnedResources],
        };
    }
}
