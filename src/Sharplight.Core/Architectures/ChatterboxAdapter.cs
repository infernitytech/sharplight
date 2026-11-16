using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Chatterbox (canonical architecture <c>chatterbox</c>, Resemble AI's text-to-speech model): T3, a Llama token language model fed with embedding rows
/// (conditioning, text with learned positions, speech tokens with learned positions), generates S3 speech tokens; S3Gen's flow turns them, after the
/// voice's prompt tokens, into a mel spectrogram (an upsampling conformer encoder for the mean, ten Euler steps of a conditional flow-matching U-Net
/// with classifier-free guidance); HiFT turns the mel into the waveform. The decoder graphs come from <see cref="DecoderAdapter"/> (the T3 weights are
/// canonical decoder weights), the others from <see cref="ChatterboxConditionGraph"/>, <see cref="ChatterboxFlowEncoderGraph"/>,
/// <see cref="ChatterboxEstimatorGraph"/> and <see cref="ChatterboxVocoderGraph"/>; <see cref="ChatterboxSynthesizer"/> runs them. The model is a
/// <see cref="ModelKinds.SpeechSynthesis"/> model whose voices are attachments (<c>voices/{name}/*</c>, the built-in one is <c>default</c>).
/// </summary>
internal sealed class ChatterboxAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "chatterbox";

    /// <summary>The role of the T3 language model component.</summary>
    public const string TextModelRole = "text-model";

    /// <summary>The role of the conditioning component.</summary>
    public const string ConditionRole = "condition";

    /// <summary>The role of the flow encoder component.</summary>
    public const string FlowEncoderRole = "flow-encoder";

    /// <summary>The role of the flow-matching estimator component.</summary>
    public const string EstimatorRole = "flow-estimator";

    /// <summary>The role of the vocoder component.</summary>
    public const string VocoderRole = "vocoder";

    /// <summary>The family recorded in the text model's configuration.</summary>
    public const string Family = "chatterbox";

    /// <summary>The prefix of the voice attachments (<c>voices/{name}/speaker.f32</c> and the other parts of a voice).</summary>
    public const string VoicePrefix = "voices/";

    /// <summary>The most speech tokens one request generates.</summary>
    public const int MaxSpeechTokens = 1000;

    /// <summary>The most flow tokens (prompt plus generated).</summary>
    public const int MaxFlowTokens = 1536;

    /// <inheritdoc/>
    public string Name => "Chatterbox";

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
        int hidden = src.Get<int>("decoder.hidden_size");
        long[] rates = src.Get<long[]>(a + "hift.upsample_rates");
        long[] kernels = src.Get<long[]>(a + "hift.upsample_kernels");
        int nFft = src.Get<int>(a + "hift.fft_size");
        int hop = src.Get<int>(a + "hift.hop");
        int harmonics = src.Get<int>(a + "hift.harmonics") + 1;
        int encoderLayers = CountLayers(src, "s3gen.flow.encoder.encoders.");
        int upLayers = CountLayers(src, "s3gen.flow.encoder.up_encoders.");
        int midBlocks = CountLayers(src, "s3gen.flow.decoder.estimator.mid_blocks.");
        int transformers = CountLayers(src, "s3gen.flow.decoder.estimator.down_blocks.0.1.");
        var weights = text.Weights;
        WeightInfo? Lookup(string name) => weights.GetValueOrDefault(name);
        var graphs = new Dictionary<string, ComputeGraph>(text.Graphs, StringComparer.Ordinal)
        {
            [ChatterboxConditionGraph.GraphName] = ChatterboxConditionGraph.Build(Lookup, src.Get<int>(a + "t3.prompt_tokens"), src.Get<int>(a + "t3.perceiver_queries"),
                src.Get<int>(a + "t3.perceiver_heads"), hidden),
            [ChatterboxFlowEncoderGraph.GraphName] = ChatterboxFlowEncoderGraph.Build(Lookup, encoderLayers, upLayers),
            [ChatterboxEstimatorGraph.GraphName] = ChatterboxEstimatorGraph.Build(Lookup, midBlocks, transformers),
            [ChatterboxVocoderGraph.GraphName] = ChatterboxVocoderGraph.Build(Lookup, rates, kernels, nFft, hop, harmonics),
        };
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
                    WeightPrefixes = ["layers.", "token_embedding.", "output_norm.", "lm_head.", "rope.", "t3.text_emb.", "t3.text_pos_emb.", "t3.speech_pos_emb."],
                    Config = ComponentConfig.Of(("family", Family), ("startText", src.Get<int>(a + "t3.start_text_token")), ("stopText", src.Get<int>(a + "t3.stop_text_token")),
                        ("startSpeech", src.Get<int>(a + "t3.start_speech_token")), ("stopSpeech", src.Get<int>(a + "t3.stop_speech_token")),
                        ("speechVocabulary", src.Get<int>(a + "t3.speech_vocabulary"))),
                },
                new ModelComponent
                {
                    Role = ConditionRole,
                    Graphs = [ChatterboxConditionGraph.GraphName],
                    WeightPrefixes = ["t3.cond_enc.", "t3.speech_pos_emb.", "token_embedding."],
                    Config = ComponentConfig.Of(("promptTokens", src.Get<int>(a + "t3.prompt_tokens"))),
                    Attachments = src.Attachments.Keys.Where(k => k.StartsWith(VoicePrefix, StringComparison.Ordinal)).ToDictionary(k => k, k => k, StringComparer.Ordinal),
                },
                new ModelComponent
                {
                    Role = FlowEncoderRole,
                    Graphs = [ChatterboxFlowEncoderGraph.GraphName],
                    WeightPrefixes = ["s3gen.flow.input_embedding.", "s3gen.flow.encoder.", "s3gen.flow.encoder_proj.", "s3gen.flow.spk_embed_affine_layer."],
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxFlowTokens },
                },
                new ModelComponent
                {
                    Role = EstimatorRole,
                    Graphs = [ChatterboxEstimatorGraph.GraphName],
                    WeightPrefixes = ["s3gen.flow.decoder."],
                    Config = ComponentConfig.Of(("steps", src.Get<int>(a + "s3gen.flow_steps")), ("guidance", src.Get<double>(a + "s3gen.guidance"))),
                    SymbolBounds = new Dictionary<string, long> { [ChatterboxEstimatorGraph.FrameSymbol] = 2L * MaxFlowTokens },
                },
                new ModelComponent
                {
                    Role = VocoderRole,
                    Graphs = [ChatterboxVocoderGraph.GraphName],
                    WeightPrefixes = ["s3gen.mel2wav."],
                    Config = ComponentConfig.Of(("samplesPerFrame", samplesPerFrame), ("fftSize", nFft), ("hop", hop), ("harmonics", harmonics)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = 2L * MaxSpeechTokens },
                },
                new ModelComponent { Role = ModelComponentRoles.Tokenizer, MetadataPrefix = null },
            ],
            Config = ComponentConfig.Of(("sampleRate", src.Get<int>(a + "sample_rate"))),
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

    // The number of indexed modules under a prefix ("prefix{i}.").
    private static int CountLayers(ModelSource src, string prefix)
    {
        int count = 0;
        while (src.Weights.Keys.Any(k => k.StartsWith($"{prefix}{count}.", StringComparison.Ordinal)))
        {
            count++;
        }

        return count;
    }
}
