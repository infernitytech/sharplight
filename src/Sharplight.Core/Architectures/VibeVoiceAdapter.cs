using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// VibeVoice text to speech (canonical architecture <see cref="Architecture"/>; Microsoft's VibeVoice-1.5B): a Qwen2 language model reads a script of speaker turns (and, for voice
/// cloning, the acoustic latents of reference recordings in the place of audio placeholder tokens) and, one frame at a time at 7.5 Hz, either ends the speech or asks the
/// diffusion head for the next acoustic latent: the head denoises a 64-wide latent conditioned on the language model's hidden state (DPM-Solver over a cosine schedule with
/// classifier-free guidance against a second, context-free run of the language model); the acoustic tokenizer's decoder turns latents into 24 kHz audio, and the next input of the
/// language model is the sum of the acoustic connector's image of the latent and the semantic connector's image of the semantic tokenizer's encoding of the audio just generated.
/// The language model is built by <see cref="DecoderAdapter"/> over embedding rows (<c>decoder.embedding_input</c>); the host-side tables (the token embedding, the latent scale and bias)
/// and the diffusion schedule stay with <see cref="VibeVoiceSynthesizer"/>.
/// </summary>
/// <remarks>
/// Hyper-parameters: the language model's <c>decoder.*</c> keys; <c>vibevoice.{audio_bos_token_id,audio_eos_token_id,audio_token_id,eos_token_id,sample_rate,hidden,latent,semantic_latent,
/// voice_frames}</c>; <c>vibevoice.diffusion.{layers,hidden,frequencies,eps,train_steps,max_period,beta_schedule,prediction_type}</c>; the tokenizers' <c>vibevoice.{acoustic,semantic}.{filters,kernel,
/// depths,ratios,latent,eps}</c> (<see cref="VibeVoiceTokenizerParameters"/>); the defaults <c>vibevoice.sampling.{cfg_scale,diffusion_steps,max_frames}</c>; the prompt texts
/// <c>vibevoice.prompt.{system,voice,text,speech}</c>. Weights: the language model's canonical decoder names; <c>{acoustic,semantic}.{encoder,decoder}.</c> (tokenizer weights, see
/// <see cref="VibeVoiceTokenizerGraph"/>: <c>down.{i}</c>, <c>up.{i}</c>, <c>stage.{i}.{j}.{norm,mixer,gamma,ffn_norm,ffn1,ffn2,ffn_gamma}</c>, <c>head</c>); <c>acoustic_connector.</c> and
/// <c>semantic_connector.</c> <c>{fc1,norm,fc2}</c>; <c>diffusion.</c> (<c>noisy_proj</c>, <c>cond_proj</c>, <c>time.{fc1,fc2}</c>, <c>layers.{i}.{ada,norm,ffn.{gate,up,down}}</c>,
/// <c>final.{ada,out}</c>); <c>vibevoice.latent_scale</c> and <c>vibevoice.latent_bias</c>.
/// </remarks>
internal sealed class VibeVoiceAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "vibevoice";

    /// <summary>The model family recorded in the language model component's configuration.</summary>
    public const string Family = "vibevoice";

    /// <summary>The role of the language model component.</summary>
    public const string TextModelRole = "text-model";

    /// <summary>The role of the acoustic tokenizer's decoder (graph <see cref="AcousticDecodeGraph"/>).</summary>
    public const string AcousticDecoderRole = "acoustic-decoder";

    /// <summary>The role of the acoustic tokenizer's encoder (graph <see cref="AcousticEncodeGraph"/>).</summary>
    public const string AcousticEncoderRole = "acoustic-encoder";

    /// <summary>The role of the semantic tokenizer's encoder (graph <see cref="SemanticEncodeGraph"/>).</summary>
    public const string SemanticEncoderRole = "semantic-encoder";

    /// <summary>The role of the diffusion head (graph <see cref="VibeVoiceDiffusionGraph.Graph"/>).</summary>
    public const string DiffusionRole = "diffusion-head";

    /// <summary>The role of the connectors (graphs <see cref="SpeechConnectorGraph"/> and <see cref="VoiceConnectorGraph"/>).</summary>
    public const string ConnectorRole = "connector";

    /// <summary>The decoder graph: one latent frame to <see cref="Hop"/> samples.</summary>
    public const string AcousticDecodeGraph = "acoustic.decode";

    /// <summary>The acoustic encoder graph: <see cref="EncodeFrames"/> frames of audio to latents.</summary>
    public const string AcousticEncodeGraph = "acoustic.encode";

    /// <summary>The semantic encoder graph: one frame of audio to its semantic latent.</summary>
    public const string SemanticEncodeGraph = "semantic.encode";

    /// <summary>The connector graph of generated frames: acoustic latents [T, latent] and semantic latents [T, semantic latent] to language model rows.</summary>
    public const string SpeechConnectorGraph = "connector.speech";

    /// <summary>The connector graph of voice prompts: acoustic latents [T, latent] to language model rows.</summary>
    public const string VoiceConnectorGraph = "connector.voice";

    /// <summary>The frames the acoustic encoder graph covers per run (longer references run in several, a shorter tail is padded with silence).</summary>
    public const int EncodeFrames = 64;

    /// <summary>The most frames of a voice prompt (rows of the connector's symbol bound).</summary>
    public const int MaxVoiceFrames = 1024;

    /// <summary>The samples one latent frame covers.</summary>
    public const int Hop = 3200;

    /// <summary>The prefix of the tokenizers' weights: acoustic.</summary>
    public const string AcousticPrefix = "acoustic.";

    /// <summary>The prefix of the semantic tokenizer's weights.</summary>
    public const string SemanticPrefix = "semantic.";

    /// <inheritdoc/>
    public string Name => "VibeVoice";

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
        var weights = new Dictionary<string, WeightInfo>(backbone.Weights, StringComparer.Ordinal);
        WeightInfo? Weight(string name) => weights.GetValueOrDefault(name);

        var acoustic = VibeVoiceTokenizerParameters.From(src, a + "acoustic.");
        var semantic = VibeVoiceTokenizerParameters.From(src, a + "semantic.");
        int hidden = src.Get<int>(a + "hidden");
        int latent = acoustic.Latent;
        var graphs = new Dictionary<string, ComputeGraph>(backbone.Graphs, StringComparer.Ordinal)
        {
            [AcousticDecodeGraph] = BuildAcousticDecode(acoustic, Weight),
            [AcousticEncodeGraph] = BuildEncode(acoustic, Weight, AcousticEncodeGraph, AcousticPrefix, EncodeFrames),
            [SemanticEncodeGraph] = BuildEncode(semantic, Weight, SemanticEncodeGraph, SemanticPrefix, 1),
            [SpeechConnectorGraph] = BuildConnector(Weight, SpeechConnectorGraph, latent, semantic.Latent, true),
            [VoiceConnectorGraph] = BuildConnector(Weight, VoiceConnectorGraph, latent, semantic.Latent, false),
            [VibeVoiceDiffusionGraph.Graph] = VibeVoiceDiffusionGraph.Build(Weight, src.Get<int>(a + "diffusion.layers"), latent, hidden, src.Get<int>(a + "diffusion.frequencies"),
                src.Get<float>(a + "diffusion.eps"), 2),
        };

        var config = new List<(string, object)>
        {
            ("family", Family), ("audioBos", src.Get<int>(a + "audio_bos_token_id")), ("audioEos", src.Get<int>(a + "audio_eos_token_id")),
            ("audioToken", src.Get<int>(a + "audio_token_id")), ("eos", src.Get<int>(a + "eos_token_id")), ("hidden", hidden), ("latent", latent),
            ("semanticLatent", semantic.Latent), ("sampleRate", src.Get<int>(a + "sample_rate")), ("hop", (int)acoustic.Hop),
            ("diffusion.trainSteps", src.Get<int>(a + "diffusion.train_steps")), ("diffusion.frequencies", src.Get<int>(a + "diffusion.frequencies")),
            ("diffusion.maxPeriod", src.Get<float>(a + "diffusion.max_period")), ("diffusion.betaSchedule", src.Get<string>(a + "diffusion.beta_schedule")),
            ("diffusion.predictionType", src.Get<string>(a + "diffusion.prediction_type")), ("normalizeDbfs", src.Get<float>(a + "normalize_dbfs")), ("vaeStd", src.Get(a + "acoustic.vae_std", 0.625f)),
            ("prompt.system", src.Get<string>(a + "prompt.system")), ("prompt.voice", src.Get<string>(a + "prompt.voice")), ("prompt.text", src.Get<string>(a + "prompt.text")),
            ("prompt.speech", src.Get<string>(a + "prompt.speech")),
        };
        foreach (var key in new[] { "cfg_scale", "diffusion_steps", "max_frames" })
        {
            if (src.Has(a + "sampling." + key))
            {
                config.Add(("sampling." + key, src.Get<float>(a + "sampling." + key)));
            }
        }

        string[] ownPrefixes = [AcousticPrefix, SemanticPrefix, "acoustic_connector.", "semantic_connector.", "diffusion.", a];
        var backbonePrefixes = backbone.Weights.Keys.Where(k => !ownPrefixes.Any(p => k.StartsWith(p, StringComparison.Ordinal)))
            .Select(k => k[..(k.IndexOf('.', StringComparison.Ordinal) + 1)]).Where(p => p.Length > 0).Distinct().Order(StringComparer.Ordinal).ToList();
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
                new ModelComponent { Role = AcousticDecoderRole, Graphs = [AcousticDecodeGraph], WeightPrefixes = [AcousticPrefix + "decoder."] },
                new ModelComponent { Role = AcousticEncoderRole, Graphs = [AcousticEncodeGraph], WeightPrefixes = [AcousticPrefix + "encoder."] },
                new ModelComponent { Role = SemanticEncoderRole, Graphs = [SemanticEncodeGraph], WeightPrefixes = [SemanticPrefix] },
                new ModelComponent { Role = DiffusionRole, Graphs = [VibeVoiceDiffusionGraph.Graph], WeightPrefixes = ["diffusion."] },
                new ModelComponent
                {
                    Role = ConnectorRole,
                    Graphs = [SpeechConnectorGraph, VoiceConnectorGraph],
                    WeightPrefixes = ["acoustic_connector.", "semantic_connector."],
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = MaxVoiceFrames },
                },
            ],
            Config = ComponentConfig.Of(("sampleRate", src.Get<int>(a + "sample_rate"))),
        };
        return new ModelDefinition
        {
            Architecture = Architecture,
            Name = backbone.Name,
            Weights = weights,
            Graphs = graphs,
            States = backbone.States,
            Properties = backbone.Properties,
            SourceMetadata = backbone.SourceMetadata,
            Description = description,
            Attachments = backbone.Attachments,
            OwnedResources = backbone.OwnedResources,
        };
    }

    /// <summary>One latent frame [1, latent] to audio [hop, 1] (with the convolutions' caches as inputs and outputs).</summary>
    private static ComputeGraph BuildAcousticDecode(VibeVoiceTokenizerParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AcousticDecodeGraph, weights);
        var latents = b.Input("latents", InputRole.Other, DataType.F32, new Shape(1, p.Latent));
        b.Output(AudioCodecGraphs.AudioOutput, VibeVoiceTokenizerGraph.Decode(b, p, latents, AcousticPrefix, AcousticPrefix));
        return b.Build();
    }

    /// <summary><paramref name="frames"/> frames of audio [frames · hop, 1] to latents [frames, latent].</summary>
    internal static ComputeGraph BuildEncode(VibeVoiceTokenizerParameters p, Func<string, WeightInfo?> weights, string name, string prefix, int frames)
    {
        var b = new GraphBuilder(name, weights);
        var audio = b.Input(AudioCodecGraphs.AudioInput, InputRole.Media, DataType.F32, new Shape(frames * p.Hop, 1));
        b.Output("latents", VibeVoiceTokenizerGraph.Encode(b, p, audio, prefix, prefix));
        return b.Build();
    }

    // The connector: Linear, RMS norm (weight), Linear.
    internal static Value Connector(GraphBuilder b, Value x, string name)
    {
        var h = b.Linear(x, name + ".fc1.weight", name + ".fc1.bias");
        h = b.RmsNorm(h, b.Weight(name + ".norm.weight"), 1e-6f);
        return b.Linear(h, name + ".fc2.weight", name + ".fc2.bias");
    }

    /// <summary>The rows the language model reads for generated frames (acoustic and semantic connectors summed) or for voice prompts (acoustic only).</summary>
    private static ComputeGraph BuildConnector(Func<string, WeightInfo?> weights, string name, int latent, int semanticLatent, bool withSemantic)
    {
        var b = new GraphBuilder(name, weights);
        var frames = Dim.Sym(WellKnown.TokenSymbol);
        var acoustic = b.Input("acoustic", InputRole.Other, DataType.F32, new Shape(frames, latent));
        var rows = Connector(b, acoustic, "acoustic_connector");
        if (withSemantic)
        {
            var semantic = b.Input("semantic", InputRole.Other, DataType.F32, new Shape(frames, semanticLatent));
            rows = b.Add(rows, Connector(b, semantic, "semantic_connector"));
        }

        b.Output("rows", rows);
        return b.Build();
    }
}
