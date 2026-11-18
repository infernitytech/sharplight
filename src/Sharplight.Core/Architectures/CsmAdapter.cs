using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// CSM (canonical architecture <see cref="Architecture"/>; Sesame's conversational speech model): a Llama backbone over text tokens and audio frames, a small Llama depth
/// decoder that predicts a frame's codebooks 1 to 31 from the backbone's hidden state and the codes so far, and the bundled Mimi codec (<see cref="MimiGraph"/>) that decodes the
/// 32 codebooks of every frame to 24 kHz audio at 12.5 frames per second. The backbone is built by <see cref="DecoderAdapter"/> over embedding rows; the depth decoder is a second
/// decoder whose weights and caches live under <c>depth.</c> (graph <see cref="DepthGraph"/>); the text embedding, the codebook embedding table (the backbone's input rows are the sum
/// of a frame's 32 entries, the depth decoder's inputs single entries), the depth decoder's embedding projector and its 31 codebook heads are read on the host.
/// <see cref="CsmSynthesizer"/> runs it.
/// </summary>
/// <remarks>
/// Hyper-parameters: the backbone's decoder keys (<c>decoder.*</c>), the depth decoder's under <c>csm.depth.decoder.*</c>, the codec's Mimi keys under <c>csm.codec.*</c>, and
/// <c>csm.{text_vocab_size,codebook_vocab_size,codebooks,audio_token_id,audio_eos_token_id,codebook_eos_token_id,codebook_pad_token_id,bos_token_id,depth_hidden_size}</c> and the sampling
/// defaults <c>csm.sampling.*</c>. Weights: the backbone's canonical decoder names, <c>depth.</c> plus the depth decoder's, <c>codec.</c> plus the Mimi canonical names, and
/// <c>csm.{text_embedding.weight,audio_embedding.weight,depth.projector.weight,depth.heads.weight}</c>.
/// </remarks>
internal sealed class CsmAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "csm";

    /// <summary>The model family recorded in the backbone component's configuration.</summary>
    public const string Family = "csm";

    /// <summary>The role of the backbone component.</summary>
    public const string TextModelRole = "text-model";

    /// <summary>The role of the depth decoder component.</summary>
    public const string DepthRole = "depth-decoder";

    /// <summary>The role of the codec decoder component.</summary>
    public const string CodecRole = "codec-decoder";

    /// <summary>The prefix of the codec's weights in the model.</summary>
    public const string CodecPrefix = "codec.";

    /// <summary>The prefix of the depth decoder's weights and states in the model.</summary>
    public const string DepthPrefix = "depth.";

    /// <summary>The depth decoder's decode graph.</summary>
    public const string DepthGraph = DepthPrefix + WellKnown.DecodeMultimodalGraph;

    /// <summary>The frames one codec decode covers.</summary>
    public const int CodecChunkFrames = 250;

    /// <summary>The frames before each codec chunk the decoder sees as context (their audio is dropped).</summary>
    public const int CodecContextFrames = 25;

    /// <inheritdoc/>
    public string Name => "CSM";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
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

        string depthKeys = a + "depth.";
        var depthMetadata = src.Metadata.Where(kv => kv.Key.StartsWith(depthKeys, StringComparison.Ordinal))
            .ToDictionary(kv => kv.Key[depthKeys.Length..], kv => kv.Value, StringComparer.Ordinal);
        depthMetadata[CanonicalKeys.Architecture] = "decoder";
        depthMetadata["decoder.embedding_input"] = true;
        var depthSource = new ModelSource
        {
            Format = src.Format,
            Architecture = "decoder",
            Metadata = depthMetadata,
            Weights = src.Weights.Where(kv => kv.Key.StartsWith(DepthPrefix, StringComparison.Ordinal))
                .ToDictionary(kv => kv.Key[DepthPrefix.Length..], kv => kv.Value, StringComparer.Ordinal),
            Path = src.Path,
        };
        var depthParameters = DecoderHyperParameters.Read(depthSource);
        var depthBuilder = new DecoderGraphBuilder(depthSource, options, depthParameters, statePrefix: DepthPrefix);
        var depthGraph = depthBuilder.BuildMultimodal(0);

        string codecKeys = a + "codec.";
        var codecSource = new ModelSource
        {
            Format = src.Format,
            Architecture = MimiParameters.Architecture,
            Metadata = src.Metadata.Where(kv => kv.Key.StartsWith(codecKeys, StringComparison.Ordinal))
                .ToDictionary(kv => MimiParameters.Architecture + "." + kv.Key[codecKeys.Length..], kv => kv.Value, StringComparer.Ordinal),
            Weights = src.Weights.Where(kv => kv.Key.StartsWith(CodecPrefix, StringComparison.Ordinal))
                .ToDictionary(kv => kv.Key[CodecPrefix.Length..], kv => kv.Value, StringComparer.Ordinal),
            Path = src.Path,
        };
        var mimi = MimiParameters.From(codecSource);
        var weights = new Dictionary<string, WeightInfo>(backbone.Weights, StringComparer.Ordinal);
        var graphs = new Dictionary<string, ComputeGraph>(backbone.Graphs, StringComparer.Ordinal)
        {
            [DepthGraph] = depthGraph,
            [AudioCodecGraphs.DecodeGraph] = MimiGraph.BuildDecoder(mimi, n => weights.GetValueOrDefault(CodecPrefix + n)),
        };
        var config = new List<(string, object)>
        {
            ("family", Family), ("codebooks", src.Get<int>(a + "codebooks")), ("codebookVocabulary", src.Get<int>(a + "codebook_vocab_size")),
            ("textVocabulary", src.Get<int>(a + "text_vocab_size")), ("audioTokenId", src.Get<int>(a + "audio_token_id")),
            ("audioEosTokenId", src.Get<int>(a + "audio_eos_token_id")), ("codebookEos", src.Get<int>(a + "codebook_eos_token_id")),
            ("codebookPad", src.Get<int>(a + "codebook_pad_token_id")), ("bosTokenId", src.Get<int>(a + "bos_token_id")),
        };
        foreach (var key in new[] { "temperature", "top_k", "top_p", "depth.temperature", "depth.top_k", "depth.top_p", "max_frames" })
        {
            if (src.Has(a + "sampling." + key))
            {
                config.Add(("sampling." + key, src.Get<float>(a + "sampling." + key)));
            }
        }

        var backbonePrefixes = backbone.Weights.Keys.Where(k => !k.StartsWith(DepthPrefix, StringComparison.Ordinal) && !k.StartsWith(CodecPrefix, StringComparison.Ordinal))
            .Select(k => k[..(k.IndexOf('.', StringComparison.Ordinal) + 1)]).Where(p => p.Length > 0).Distinct().Order(StringComparer.Ordinal).ToList();
        var common = new (string, object)[]
        {
            (AudioCodecGraphs.SampleRateKey, mimi.SampleRate), (AudioCodecGraphs.SamplesPerFrameKey, mimi.SamplesPerFrame), (AudioCodecGraphs.CodebooksKey, mimi.Codebooks),
            (AudioCodecGraphs.DefaultCodebooksKey, mimi.Codebooks), (AudioCodecGraphs.CodebookSizeKey, mimi.CodebookSize),
        };
        var description = new ModelDescription
        {
            Kind = ModelKinds.SpeechSynthesis,
            Components =
            [
                new ModelComponent
                {
                    Role = TextModelRole,
                    Graphs = [.. backbone.Graphs.Keys.Order(StringComparer.Ordinal)],
                    WeightPrefixes = backbonePrefixes,
                    Config = ComponentConfig.Of([.. config]),
                },
                new ModelComponent { Role = ModelComponentRoles.Tokenizer, MetadataPrefix = null },
                new ModelComponent
                {
                    Role = DepthRole,
                    Graphs = [DepthGraph],
                    WeightPrefixes = [DepthPrefix],
                    Config = ComponentConfig.Of(("hidden", depthParameters.NEmbd)),
                },
                new ModelComponent
                {
                    Role = CodecRole,
                    Graphs = [AudioCodecGraphs.DecodeGraph],
                    WeightPrefixes = [CodecPrefix],
                    Config = ComponentConfig.Of([.. common, (AudioCodecGraphs.PositionsNumeratorKey, (int)mimi.LatentStride), (AudioCodecGraphs.PositionsDenominatorKey, 1)]),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = CodecChunkFrames + CodecContextFrames },
                },
            ],
            Config = ComponentConfig.Of(("sampleRate", mimi.SampleRate)),
        };
        return new ModelDefinition
        {
            Architecture = Architecture,
            Name = backbone.Name,
            Weights = weights,
            Graphs = graphs,
            States = [.. backbone.States, .. depthBuilder.States],
            Properties = backbone.Properties,
            SourceMetadata = backbone.SourceMetadata,
            Description = description,
            Attachments = backbone.Attachments,
            OwnedResources = backbone.OwnedResources,
        };
    }
}
