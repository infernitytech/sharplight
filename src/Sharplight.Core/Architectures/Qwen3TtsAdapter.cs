using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Qwen3-TTS (canonical architecture <see cref="Architecture"/>) with its 12 Hz codec: a composite adapter (<see cref="ICompositeModelAdapter"/>) whose primary source is
/// the TTS model and whose <see cref="AuxiliaryRoles.Vocoder"/> source is the codec (<see cref="Qwen3TtsCodecAdapter"/>'s architecture, weights under <c>codec.</c>).
/// The talker (a Qwen3 decoder over the codec's first codebook, fed with embedding rows) is built by <see cref="DecoderAdapter"/>; the code predictor (a small Qwen3
/// decoder that predicts codebooks 1 to 15 of each frame from the talker's hidden state) is a second decoder whose weights and caches live under <c>code_predictor.</c>
/// (graph <see cref="CodePredictorGraph"/>); the codec's decoder graph turns the 16 codebooks into 24 kHz audio. The text embedding, the text projection and the code
/// predictor's codebook embeddings and heads are read on the host. <see cref="Qwen3TtsSynthesizer"/> runs it. A voice-cloning model also has an ECAPA-TDNN speaker
/// encoder (<see cref="EcapaTdnnGraph"/>, weights under <see cref="SpeakerEncoderPrefix"/>) with its log-mel front end, and the codec's encoder graph (the reference
/// recording's codes for the in-context prompt).
/// </summary>
/// <remarks>
/// Hyper-parameters: the talker's decoder keys (<c>decoder.*</c>), the code predictor's under <c>qwen3_tts.code_predictor.decoder.*</c>, and <c>qwen3_tts.{text_bos,text_eos,
/// text_pad,text_hidden_size,code_groups,codec_bos,codec_eos,codec_pad,codec_think,codec_nothink,codec_think_bos,codec_think_eos,languages,language_ids,speakers,speaker_ids,
/// speaker_dialects,instructions,interleaved_text}</c>, the speaker encoder's <c>qwen3_tts.speaker_encoder.{mel_bins,embedding,channels,kernels,dilations,
/// attention_channels,res2net_scale,se_channels,sample_rate,fft_size,hop_length}</c> (with the attachment <c>speaker-features/mel-filters</c>) and the sampling defaults <c>qwen3_tts.sampling.*</c>. Weights: the talker's canonical decoder names, <c>code_predictor.</c> plus the code predictor's,
/// and <c>qwen3_tts.{text_embedding.weight,text_projection.linear_fc{1,2}.{weight,bias},code_predictor.embedding.{i}.weight,code_predictor.head.{i}.weight,
/// code_predictor.projection.{weight,bias}}</c>; a voice-cloning model's speaker encoder under <c>speaker_encoder.</c> (<see cref="EcapaTdnnGraph"/>).
/// </remarks>
internal sealed class Qwen3TtsAdapter : ICompositeModelAdapter
{
    /// <summary>The canonical architecture id of the TTS model.</summary>
    public const string Architecture = "qwen3_tts";

    /// <summary>The role of the talker component.</summary>
    public const string TextModelRole = "text-model";

    /// <summary>The role of the code predictor component.</summary>
    public const string CodePredictorRole = "code-predictor";

    /// <summary>The role of the codec decoder component.</summary>
    public const string CodecRole = "codec-decoder";

    /// <summary>The role of the codec encoder component (voice-cloning models).</summary>
    public const string CodecEncoderRole = "codec-encoder";

    /// <summary>The role of the speaker encoder component (voice-cloning models).</summary>
    public const string SpeakerEncoderRole = "speaker-encoder";

    /// <summary>The role of the speaker encoder's log-mel front end.</summary>
    public const string SpeakerFeaturesRole = "speaker-features";

    /// <summary>The prefix of the speaker encoder's weights.</summary>
    public const string SpeakerEncoderPrefix = "speaker_encoder.";

    /// <summary>The prefix of the codec's weights in the model.</summary>
    public const string CodecPrefix = "codec.";

    /// <summary>The prefix of the code predictor's weights and states in the model.</summary>
    public const string CodePredictorPrefix = "code_predictor.";

    /// <summary>The code predictor's decode graph.</summary>
    public const string CodePredictorGraph = CodePredictorPrefix + WellKnown.DecodeMultimodalGraph;

    /// <summary>The model family recorded in the talker component's configuration.</summary>
    public const string Family = "qwen3_tts";

    /// <summary>The frames one codec decode covers (the reference decodes in chunks of this many).</summary>
    public const int CodecChunkFrames = 300;

    /// <summary>The frames before each codec chunk the decoder sees as context (their audio is dropped).</summary>
    public const int CodecContextFrames = 25;

    /// <summary>The most frames one request generates (about 5.5 minutes of speech).</summary>
    public const int MaxFrames = 4096;

    /// <summary>The longest reference recording the speaker encoder reads, in codec frames (the rest of a longer recording is ignored).</summary>
    public const int MaxReferenceFrames = 750;

    /// <inheritdoc/>
    public string Name => "Qwen3-TTS";

    /// <inheritdoc/>
    public bool CanLoad(ModelSourceSet sources)
        => sources.Primary.Architecture == Architecture && sources.Auxiliary.TryGetValue(AuxiliaryRoles.Vocoder, out var codec)
           && codec.Architecture == Qwen3TtsCodecParameters.Architecture;

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options, ArchitectureCatalog architectures)
    {
        var src = sources.Primary;
        var codecSource = sources.Auxiliary[AuxiliaryRoles.Vocoder];
        string a = Architecture + ".";
        var talkerSource = new ModelSource
        {
            Format = src.Format,
            Architecture = "decoder",
            Metadata = new Dictionary<string, object>(src.Metadata, StringComparer.Ordinal) { ["decoder.embedding_input"] = true },
            Weights = src.Weights,
            Attachments = src.Attachments,
            Path = src.Path,
        };
        var talker = new DecoderAdapter().Build(new ModelSourceSet { Primary = talkerSource }, options);

        string predictorKeys = a + "code_predictor.";
        var predictorMetadata = src.Metadata.Where(kv => kv.Key.StartsWith(predictorKeys, StringComparison.Ordinal))
            .ToDictionary(kv => kv.Key[predictorKeys.Length..], kv => kv.Value, StringComparer.Ordinal);
        predictorMetadata[CanonicalKeys.Architecture] = "decoder";
        predictorMetadata["decoder.embedding_input"] = true;
        var predictorSource = new ModelSource
        {
            Format = src.Format,
            Architecture = "decoder",
            Metadata = predictorMetadata,
            Weights = src.Weights.Where(kv => kv.Key.StartsWith(CodePredictorPrefix, StringComparison.Ordinal))
                .ToDictionary(kv => kv.Key[CodePredictorPrefix.Length..], kv => kv.Value, StringComparer.Ordinal),
            Path = src.Path,
        };
        var predictorParameters = DecoderHyperParameters.Read(predictorSource);
        var predictorBuilder = new DecoderGraphBuilder(predictorSource, options, predictorParameters, statePrefix: CodePredictorPrefix);
        var predictorGraph = predictorBuilder.BuildMultimodal(0);

        var weights = new Dictionary<string, WeightInfo>(talker.Weights, StringComparer.Ordinal);
        foreach (var (name, weight) in codecSource.Weights)
        {
            weights[CodecPrefix + name] = weight.As(CodecPrefix + name);
        }

        var codec = Qwen3TtsCodecParameters.From(codecSource);
        WeightInfo? Codec(string n) => weights.GetValueOrDefault(CodecPrefix + n);
        var graphs = new Dictionary<string, ComputeGraph>(talker.Graphs, StringComparer.Ordinal)
        {
            [CodePredictorGraph] = predictorGraph,
            [AudioCodecGraphs.DecodeGraph] = Qwen3TtsCodecGraph.BuildDecoder(codec, Codec),
        };
        var cloning = CloningComponents(src, codecSource, graphs, Codec, n => weights.GetValueOrDefault(n));
        int sampleRate = codecSource.Get<int>(Qwen3TtsCodecParameters.Architecture + ".sample_rate");
        var speakers = src.Get<string[]>(a + "speakers");
        var speakerIds = src.Get<int[]>(a + "speaker_ids");
        var dialects = src.Get<string[]>(a + "speaker_dialects");
        var languages = src.Get<string[]>(a + "languages");
        var languageIds = src.Get<int[]>(a + "language_ids");
        var config = new List<(string, object)>
        {
            ("family", Family), ("textBos", src.Get<int>(a + "text_bos")), ("textEos", src.Get<int>(a + "text_eos")), ("textPad", src.Get<int>(a + "text_pad")),
            ("codeGroups", src.Get<int>(a + "code_groups")), ("codecBos", src.Get<int>(a + "codec_bos")), ("codecEos", src.Get<int>(a + "codec_eos")),
            ("codecPad", src.Get<int>(a + "codec_pad")), ("codecThink", src.Get<int>(a + "codec_think")), ("codecNothink", src.Get<int>(a + "codec_nothink")),
            ("codecThinkBos", src.Get<int>(a + "codec_think_bos")), ("codecThinkEos", src.Get<int>(a + "codec_think_eos")),
            ("speakers", string.Join(',', speakers.Select((s, i) => $"{s}={speakerIds[i]}={dialects[i]}"))),
            ("languages", string.Join(',', languages.Select((l, i) => $"{l}={languageIds[i]}"))),
            ("instructions", src.Get(a + "instructions", false)), ("interleavedText", src.Get(a + "interleaved_text", false)),
        };
        foreach (var key in new[] { "temperature", "top_k", "top_p", "repetition_penalty", "code_predictor.temperature", "code_predictor.top_k", "code_predictor.top_p" })
        {
            if (src.Has(a + "sampling." + key))
            {
                config.Add(("sampling." + key, src.Get<float>(a + "sampling." + key)));
            }
        }

        var talkerPrefixes = talker.Weights.Keys.Where(k => !k.StartsWith(CodePredictorPrefix, StringComparison.Ordinal) && !k.StartsWith(SpeakerEncoderPrefix, StringComparison.Ordinal))
            .Select(k => k[..(k.IndexOf('.', StringComparison.Ordinal) + 1)]).Where(p => p.Length > 0).Distinct().Order(StringComparer.Ordinal).ToList();
        var description = new ModelDescription
        {
            Kind = ModelKinds.SpeechSynthesis,
            Components =
            [
                new ModelComponent
                {
                    Role = TextModelRole,
                    Graphs = [.. talker.Graphs.Keys.Order(StringComparer.Ordinal)],
                    WeightPrefixes = talkerPrefixes,
                    Config = ComponentConfig.Of([.. config]),
                },
                new ModelComponent { Role = ModelComponentRoles.Tokenizer, MetadataPrefix = null },
                new ModelComponent
                {
                    Role = CodePredictorRole,
                    Graphs = [CodePredictorGraph],
                    WeightPrefixes = [CodePredictorPrefix],
                    Config = ComponentConfig.Of(("hidden", predictorParameters.NEmbd)),
                },
                new ModelComponent
                {
                    Role = CodecRole,
                    Graphs = [AudioCodecGraphs.DecodeGraph],
                    WeightPrefixes = [CodecPrefix],
                    Config = ComponentConfig.Of(("architecture", Qwen3TtsCodecParameters.Architecture), ("sampleRate", sampleRate), ("codebooks", codec.Quantizers),
                        ("codebookSize", codec.CodebookSize), ("samplesPerFrame", codec.SamplesPerFrame)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = CodecChunkFrames + CodecContextFrames },
                },
                .. cloning,
            ],
            Config = ComponentConfig.Of(("sampleRate", sampleRate)),
        };
        var properties = new Dictionary<string, object>(talker.Properties) { [WellKnown.ContextLength] = MaxFrames + 512 };
        return new ModelDefinition
        {
            Architecture = Architecture,
            Name = talker.Name,
            Weights = weights,
            Graphs = graphs,
            States = [.. talker.States, .. predictorBuilder.States],
            Properties = properties,
            SourceMetadata = talker.SourceMetadata,
            Description = description,
            Attachments = cloning.Count == 0 ? talker.Attachments : new Dictionary<string, IWeightSource>(src.Attachments.Concat(talker.Attachments), StringComparer.Ordinal),
            OwnedResources = [.. talker.OwnedResources, .. codecSource.OwnedResources],
        };
    }

    // A voice-cloning model's speaker encoder, its log-mel front end and the codec's encoder (graphs added to graphs); none for the other models.
    private static List<ModelComponent> CloningComponents(ModelSource src, ModelSource codecSource, Dictionary<string, ComputeGraph> graphs, Func<string, WeightInfo?> codec,
        Func<string, WeightInfo?> weights)
    {
        if (!src.Weights.ContainsKey(SpeakerEncoderPrefix + "fc.weight"))
        {
            return [];
        }

        string k = Architecture + ".speaker_encoder.";
        var parameters = new EcapaTdnnParameters(src.Get<int>(k + "mel_bins"), src.Get<int>(k + "embedding"), src.Get<int[]>(k + "channels"), src.Get<int[]>(k + "kernels"),
            src.Get<int[]>(k + "dilations"), src.Get<int>(k + "attention_channels"), src.Get<int>(k + "res2net_scale"), src.Get<int>(k + "se_channels"));
        graphs[EcapaTdnnGraph.GraphName] = EcapaTdnnGraph.Build(parameters, weights, SpeakerEncoderPrefix);
        var mimi = MimiParameters.From(codecSource);
        graphs[AudioCodecGraphs.EncodeGraph] = MimiGraph.BuildEncoder(mimi, codec);
        int sampleRate = src.Get<int>(k + "sample_rate");
        return
        [
            new ModelComponent
            {
                Role = SpeakerEncoderRole,
                Graphs = [EcapaTdnnGraph.GraphName],
                WeightPrefixes = [SpeakerEncoderPrefix],
                Config = ComponentConfig.Of(("embedding", parameters.Embedding), ("sampleRate", sampleRate)),
                SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxReferenceFrames * mimi.SamplesPerFrame / src.Get<int>(k + "hop_length") + 1 },
            },
            new ModelComponent
            {
                Role = SpeakerFeaturesRole,
                Config = ComponentConfig.Of(("type", "log-mel"), ("sampleRate", sampleRate), ("fftSize", src.Get<int>(k + "fft_size")), ("hopLength", src.Get<int>(k + "hop_length")),
                    ("melBins", parameters.MelBins), ("center", false), ("edgePadding", (src.Get<int>(k + "fft_size") - src.Get<int>(k + "hop_length")) / 2), ("magnitude", true),
                    ("magnitudeGuard", 1e-9), ("logBase", "e"), ("logFloor", 1e-5)),
                Attachments = new Dictionary<string, string>(StringComparer.Ordinal) { [ModelComponentAttachments.MelFilters] = SpeakerFeaturesRole + "/mel-filters" },
            },
            new ModelComponent
            {
                Role = CodecEncoderRole,
                Graphs = [AudioCodecGraphs.EncodeGraph],
                WeightPrefixes = [CodecPrefix + "encoder", CodecPrefix + "downsample.", CodecPrefix + "quantizer."],
                Config = ComponentConfig.Of((AudioCodecGraphs.SampleRateKey, mimi.SampleRate), (AudioCodecGraphs.SamplesPerFrameKey, mimi.SamplesPerFrame),
                    (AudioCodecGraphs.CodebooksKey, mimi.Codebooks), (AudioCodecGraphs.PositionsNumeratorKey, 1), (AudioCodecGraphs.PositionsDenominatorKey, mimi.SamplesPerConvFrame)),
                SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = Qwen3TtsCodecAdapter.MaxFrames * mimi.SamplesPerFrame },
            },
        ];
    }
}
