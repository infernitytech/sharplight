using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// CosyVoice 3 (canonical architecture <c>cosyvoice3</c>, FunAudioLLM's Fun-CosyVoice3): a Qwen2 language model fed with embedding rows (start of
/// sequence, the prompt and text tokens, the task row, the voice's prompt speech tokens) generates speech tokens of the S3 tokenizer v3 (25 per
/// second, 6561 codes); the DiT flow turns them, after the voice's prompt tokens, into a mel spectrogram (ten Euler steps with classifier-free
/// guidance; the prompt's mel as condition); the causal HiFT vocoder turns the mel into 24 kHz audio. The decoder graphs come from
/// <see cref="DecoderAdapter"/>, the others from <see cref="CosyVoiceFlowGraph"/> and <see cref="CosyVoiceVocoderGraph"/>; the voice of a recording
/// comes from the S3 tokenizer graph, <see cref="CamPlusGraph"/> and the prompt log-mel (<see cref="S3SpeechEncoders"/>).
/// <see cref="CosyVoiceSynthesizer"/> runs them.
/// </summary>
internal sealed class CosyVoiceAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "cosyvoice3";

    /// <summary>The family recorded in the text model's configuration.</summary>
    public const string Family = "cosyvoice3";

    /// <summary>The role of the language model component.</summary>
    public const string TextModelRole = "text-model";

    /// <summary>The role of the flow encoder component.</summary>
    public const string FlowEncoderRole = "flow-encoder";

    /// <summary>The role of the flow-matching estimator component.</summary>
    public const string EstimatorRole = "flow-estimator";

    /// <summary>The role of the vocoder component.</summary>
    public const string VocoderRole = "vocoder";

    /// <summary>The longest recording the cloning encoders accept, in seconds.</summary>
    public const int MaxCloneSeconds = 30;

    /// <summary>The most speech tokens one request generates.</summary>
    public const int MaxSpeechTokens = 1500;

    /// <summary>The most flow tokens (prompt plus generated).</summary>
    public const int MaxFlowTokens = 2048;

    /// <summary>The weight name of the text embedding (read on the host).</summary>
    public const string TextEmbeddingWeight = Architecture + ".text_embedding";

    /// <inheritdoc/>
    public string Name => "CosyVoice";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var metadata = new Dictionary<string, object>(src.Metadata, StringComparer.Ordinal) { ["decoder.embedding_input"] = true };
        var decoderSource = new ModelSource
        {
            Format = src.Format,
            Architecture = "decoder",
            Metadata = metadata,
            Weights = src.Weights,
            Attachments = src.Attachments,
            Path = src.Path,
        };
        var text = new DecoderAdapter().Build(new ModelSourceSet { Primary = decoderSource }, options);
        string a = Architecture + ".";
        long[] rates = src.Get<long[]>(a + "hift.upsample_rates");
        long[] kernels = src.Get<long[]>(a + "hift.upsample_kernels");
        int nFft = src.Get<int>(a + "hift.fft_size");
        int hop = src.Get<int>(a + "hift.hop");
        int harmonics = src.Get<int>(a + "hift.harmonics") + 1;
        int sampleRate = src.Get<int>(a + "sample_rate");
        int mels = src.Get<int>(a + "flow.mel_bins");
        int tokenMelRatio = src.Get<int>(a + "flow.token_mel_ratio");
        var weights = text.Weights;
        WeightInfo? Lookup(string name) => weights.GetValueOrDefault(name);
        var graphs = new Dictionary<string, ComputeGraph>(text.Graphs, StringComparer.Ordinal)
        {
            [CosyVoiceFlowGraph.EncodeGraph] = CosyVoiceFlowGraph.BuildEncode(Lookup, src.Get<int>(a + "flow.pre_lookahead"), tokenMelRatio),
            [CosyVoiceFlowGraph.EstimatorGraph] = CosyVoiceFlowGraph.BuildEstimator(Lookup, src.Get<int>(a + "flow.depth"), src.Get<int>(a + "flow.heads"), mels),
            [CosyVoiceVocoderGraph.GraphName] = CosyVoiceVocoderGraph.Build(Lookup, rates, kernels, nFft, hop, harmonics, src.Get<int>(a + "hift.look_right"), sampleRate),
        };
        bool cloning = weights.ContainsKey("speech_tokenizer.encoder.conv1.weight") && weights.ContainsKey("campplus.head.conv1.weight")
            && src.Attachments.ContainsKey(S3SpeechEncoders.SpeechTokenFeaturesRole + "/mel-filters");
        if (cloning)
        {
            int layers = 0;
            while (weights.ContainsKey($"speech_tokenizer.encoder.blocks.{layers}.attn.query.weight"))
            {
                layers++;
            }

            int width = (int)weights["speech_tokenizer.encoder.conv1.weight"].Shape[0].Fixed;
            graphs[S3TokenizerGraph.GraphName] = S3TokenizerGraph.Build(Lookup, "speech_tokenizer.", 128, layers, 20, width);
            graphs[CamPlusGraph.GraphName] = CamPlusGraph.Build(Lookup, "campplus.", 80);
        }

        long samplesPerFrame = ChatterboxVocoderGraph.SamplesPerFrame(rates, hop);
        var description = new ModelDescription
        {
            Kind = ModelKinds.SpeechSynthesis,
            Components =
            [
                new ModelComponent
                {
                    Role = TextModelRole,
                    Graphs = [.. text.Graphs.Keys.Order(StringComparer.Ordinal)],
                    WeightPrefixes = ["layers.", "token_embedding.", "output_norm.", "lm_head.", "rope.", TextEmbeddingWeight],
                    Config = ComponentConfig.Of(("family", Family), ("speechTokens", src.Get<int>(a + "llm.speech_tokens")), ("sos", src.Get<int>(a + "llm.sos")),
                        ("eos", src.Get<int>(a + "llm.eos")), ("taskId", src.Get<int>(a + "llm.task_id")), ("topP", src.Get<double>(a + "llm.top_p")),
                        ("topK", src.Get<int>(a + "llm.top_k")), ("window", src.Get<int>(a + "llm.window")), ("tauR", src.Get<double>(a + "llm.tau_r")),
                        ("minTokenTextRatio", src.Get<double>(a + "llm.min_token_text_ratio")), ("maxTokenTextRatio", src.Get<double>(a + "llm.max_token_text_ratio"))),
                },
                new ModelComponent
                {
                    Role = FlowEncoderRole,
                    Graphs = [CosyVoiceFlowGraph.EncodeGraph],
                    WeightPrefixes = ["flow.input_embedding.", "flow.pre_lookahead_layer.", "flow.spk_embed_affine_layer."],
                    Config = ComponentConfig.Of(("tokenMelRatio", tokenMelRatio), ("melBins", mels)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxFlowTokens },
                },
                new ModelComponent
                {
                    Role = EstimatorRole,
                    Graphs = [CosyVoiceFlowGraph.EstimatorGraph],
                    WeightPrefixes = ["flow.decoder."],
                    Config = ComponentConfig.Of(("steps", src.Get<int>(a + "flow.steps")), ("guidance", src.Get<double>(a + "flow.guidance"))),
                    SymbolBounds = new Dictionary<string, long> { [CosyVoiceFlowGraph.FrameSymbol] = (long)tokenMelRatio * MaxFlowTokens },
                },
                new ModelComponent
                {
                    Role = VocoderRole,
                    Graphs = [CosyVoiceVocoderGraph.GraphName],
                    WeightPrefixes = ["hift."],
                    Config = ComponentConfig.Of(("samplesPerFrame", samplesPerFrame), ("fftSize", nFft), ("hop", hop), ("harmonics", harmonics)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = (long)tokenMelRatio * MaxSpeechTokens },
                },
                new ModelComponent { Role = ModelComponentRoles.Tokenizer, MetadataPrefix = null },
                .. (cloning ? CloningComponents() : []),
            ],
            Config = ComponentConfig.Of(("sampleRate", sampleRate)),
        };
        return new ModelDefinition
        {
            Architecture = Architecture,
            Name = text.Name,
            Weights = weights,
            Graphs = graphs,
            States = text.States,
            Properties = text.Properties,
            SourceMetadata = text.SourceMetadata,
            Description = description,
            Attachments = src.Attachments,
            OwnedResources = src.OwnedResources,
        };
    }

    // The encoders and front ends that turn a recording into a voice (CosyVoiceFrontEnd.frontend_zero_shot).
    private static ModelComponent[] CloningComponents()
    {
        const int Rate16 = 16000;
        long maxFrames16 = MaxCloneSeconds * Rate16 / 160;
        static Dictionary<string, string> Filters(string role)
            => new(StringComparer.Ordinal) { [ModelComponentAttachments.MelFilters] = role + "/mel-filters" };
        return
        [
            new ModelComponent
            {
                Role = S3SpeechEncoders.SpeechTokenizerRole,
                Graphs = [S3TokenizerGraph.GraphName],
                WeightPrefixes = ["speech_tokenizer."],
                Config = ComponentConfig.Of(("framesPerToken", 4), ("levels", 3)),
                SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = maxFrames16 },
            },
            new ModelComponent
            {
                Role = S3SpeechEncoders.SpeakerEncoderRole,
                Graphs = [CamPlusGraph.GraphName],
                WeightPrefixes = ["campplus."],
                SymbolBounds = new Dictionary<string, long>
                {
                    [WellKnown.TokenSymbol] = maxFrames16,
                    [CamPlusGraph.SegmentSymbol] = (maxFrames16 / 2 / CamPlusGraph.SegmentLength) + 2,
                },
            },
            new ModelComponent
            {
                Role = S3SpeechEncoders.SpeechTokenFeaturesRole,
                Config = ComponentConfig.Of(("type", "log-mel"), ("sampleRate", Rate16), ("fftSize", 400), ("hopLength", 160), ("melBins", 128),
                    ("dropLastFrame", true), ("logBase", "10"), ("logFloor", 1e-10), ("dynamicRange", 8.0), ("logOffset", 4.0), ("logScale", 0.25)),
                Attachments = Filters(S3SpeechEncoders.SpeechTokenFeaturesRole),
            },
            new ModelComponent
            {
                Role = S3SpeechEncoders.SpeakerFeaturesRole,
                Config = ComponentConfig.Of(("type", "kaldi-fbank"), ("sampleRate", Rate16), ("melBins", 80), ("subtractMean", true)),
            },
            new ModelComponent
            {
                Role = S3SpeechEncoders.PromptFeaturesRole,
                Config = ComponentConfig.Of(("type", "log-mel"), ("sampleRate", 24000), ("fftSize", 1920), ("hopLength", 480), ("melBins", 80), ("center", false),
                    ("edgePadding", 720), ("magnitude", true), ("magnitudeGuard", 1e-9), ("logBase", "e"), ("logFloor", 1e-5)),
                Attachments = Filters(S3SpeechEncoders.PromptFeaturesRole),
            },
        ];
    }
}
