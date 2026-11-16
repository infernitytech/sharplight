using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// XTTS v2 (canonical architecture <c>xtts</c>, Coqui's multilingual text-to-speech model): a GPT-2 fed with embedding rows (the speaker's conditioning
/// latents, the text with learned positions, the audio codes with learned positions) generates audio codes; its hidden states (after both output norms)
/// at the codes are the latents the HiFi-GAN decoder turns into 24 kHz audio, conditioned on the speaker embedding. The decoder graphs come from
/// <see cref="DecoderAdapter"/> (canonical GPT-2 weights, <c>output_norm_2</c> for XTTS's final norm), the waveform decoder from
/// <see cref="XttsVocoderGraph"/>; <see cref="XttsSynthesizer"/> runs them. The speakers are attachments (<c>voices/{name}/latents.f32</c>, <c>speaker.f32</c>).
/// </summary>
internal sealed class XttsAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "xtts";

    /// <summary>The role of the GPT component.</summary>
    public const string TextModelRole = "text-model";

    /// <summary>The role of the waveform decoder component.</summary>
    public const string VocoderRole = "vocoder";

    /// <summary>The role of the style encoder (the GPT's conditioning latents of a recording).</summary>
    public const string StyleEncoderRole = "style-encoder";

    /// <summary>The role of the speaker encoder (the decoder's speaker embedding of a recording).</summary>
    public const string SpeakerEncoderRole = "speaker-encoder";

    /// <summary>The role of the style encoder's front end (log-mel at 22.05 kHz).</summary>
    public const string StyleFeaturesRole = "style-features";

    /// <summary>The role of the speaker encoder's front end (log-mel at 16 kHz).</summary>
    public const string SpeakerFeaturesRole = "speaker-features";

    /// <summary>The family recorded in the text model's configuration.</summary>
    public const string Family = "xtts";

    /// <summary>The prefix of the voice attachments.</summary>
    public const string VoicePrefix = "voices/";

    /// <inheritdoc/>
    public string Name => "XTTS v2";

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
        int width = src.Get<int>("decoder.hidden_size");
        int speakerWidth = src.Get<int>(a + "speaker_width");
        var weights = text.Weights;
        var graphs = new Dictionary<string, ComputeGraph>(text.Graphs, StringComparer.Ordinal)
        {
            [XttsVocoderGraph.GraphName] = XttsVocoderGraph.Build(weights.GetValueOrDefault, width, speakerWidth, [8, 8, 2, 2], [16, 16, 4, 4]),
        };
        bool cloning = weights.ContainsKey("gpt.conditioning_encoder.init.weight") && weights.ContainsKey("hifigan_decoder.speaker_encoder.fc.weight")
            && weights.ContainsKey("gpt.mel_stats") && src.Attachments.ContainsKey(StyleFeaturesRole + "/mel-filters");
        if (cloning)
        {
            graphs[XttsConditioningGraph.GraphName] = XttsConditioningGraph.Build(weights.GetValueOrDefault, width, 6, 16, 2, 8);
            graphs[XttsSpeakerEncoderGraph.GraphName] = XttsSpeakerEncoderGraph.Build(weights.GetValueOrDefault, 64);
        }

        int maxAudio = src.Get<int>(a + "max_audio_tokens");
        int frames = (int)Math.Ceiling(maxAudio * (src.Get<int>(a + "code_stride") / (double)src.Get<int>(a + "output_hop"))
            * (src.Get<int>(a + "sample_rate") / (double)src.Get<int>(a + "input_sample_rate"))) + 8;
        var description = new ModelDescription
        {
            Kind = ModelKinds.SpeechSynthesis,
            Components =
            [
                new ModelComponent
                {
                    Role = TextModelRole,
                    Graphs = [.. text.Graphs.Keys.Order(StringComparer.Ordinal)],
                    WeightPrefixes = ["layers.", "token_embedding.", "output_norm.", "output_norm_2.", "lm_head.", "gpt."],
                    Config = ComponentConfig.Of(("family", Family), ("startText", src.Get<int>(a + "start_text_token")), ("stopText", src.Get<int>(a + "stop_text_token")),
                        ("startAudio", src.Get<int>(a + "start_audio_token")), ("stopAudio", src.Get<int>(a + "stop_audio_token")), ("maxAudioTokens", maxAudio),
                        ("codeStride", src.Get<int>(a + "code_stride")), ("outputHop", src.Get<int>(a + "output_hop")),
                        ("inputSampleRate", src.Get<int>(a + "input_sample_rate")), ("temperature", src.Get<float>(a + "temperature")),
                        ("topK", src.Get<int>(a + "top_k")), ("topP", src.Get<float>(a + "top_p")), ("repetitionPenalty", src.Get<float>(a + "repetition_penalty"))),
                    Attachments = src.Attachments.Keys.Where(k => k.StartsWith(VoicePrefix, StringComparison.Ordinal)).ToDictionary(k => k, k => k, StringComparer.Ordinal),
                },
                new ModelComponent
                {
                    Role = VocoderRole,
                    Graphs = [XttsVocoderGraph.GraphName],
                    WeightPrefixes = ["hifigan_decoder."],
                    Config = ComponentConfig.Of(("samplesPerFrame", 256), ("speakerWidth", speakerWidth)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = frames },
                },
                new ModelComponent { Role = ModelComponentRoles.Tokenizer, MetadataPrefix = null },
                .. (cloning ? CloningComponents() : []),
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

    // The encoders and front ends that turn a recording into a voice (Xtts.get_conditioning_latents with the model's settings).
    private static ModelComponent[] CloningComponents()
    {
        static Dictionary<string, string> Filters(string role)
            => new(StringComparer.Ordinal) { [ModelComponentAttachments.MelFilters] = role + "/mel-filters" };
        return
        [
            new ModelComponent
            {
                Role = StyleEncoderRole,
                Graphs = [XttsConditioningGraph.GraphName],
                WeightPrefixes = ["gpt.conditioning_encoder.", "gpt.conditioning_perceiver.", "gpt.mel_stats"],
                Config = ComponentConfig.Of(("maxSeconds", 30), ("chunkSeconds", 4)),
                SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = (4 * 22050 / 256) + 2 },
            },
            new ModelComponent
            {
                Role = SpeakerEncoderRole,
                Graphs = [XttsSpeakerEncoderGraph.GraphName],
                WeightPrefixes = ["hifigan_decoder.speaker_encoder."],
                SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = (30 * 16000 / 160) + 2 },
            },
            new ModelComponent
            {
                Role = StyleFeaturesRole,
                Config = ComponentConfig.Of(("type", "log-mel"), ("sampleRate", 22050), ("fftSize", 2048), ("windowSize", 1024), ("hopLength", 256), ("melBins", 80),
                    ("logBase", "e"), ("logFloor", 1e-5)),
                Attachments = Filters(StyleFeaturesRole),
            },
            new ModelComponent
            {
                Role = SpeakerFeaturesRole,
                Config = ComponentConfig.Of(("type", "log-mel"), ("sampleRate", 16000), ("fftSize", 512), ("windowSize", 400), ("hopLength", 160), ("melBins", 64),
                    ("window", "hamming"), ("preemphasis", 0.97), ("preemphasisEdge", "reflect"), ("logBase", "e"), ("logGuard", 1e-6)),
                Attachments = Filters(SpeakerFeaturesRole),
            },
        ];
    }
}
