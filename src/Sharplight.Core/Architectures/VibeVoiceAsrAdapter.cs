using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// VibeVoice speech recognition (canonical architecture <see cref="Architecture"/>; Microsoft's VibeVoice-ASR): the acoustic and the semantic speech tokenizer (the same encoders
/// <see cref="VibeVoiceAdapter"/> uses, <see cref="VibeVoiceTokenizerGraph"/>) turn 24 kHz audio into one 64-wide and one 128-wide latent per 3200 samples (7.5 per second); the
/// two connectors project them to the width of a Qwen2 language model and their sum replaces the audio placeholder tokens of a chat prompt; the language model then writes the
/// transcript as a JSON array of segments with start and end times, speaker numbers and content (long-form: one pass over up to an hour of audio). The language model is built by
/// <see cref="DecoderAdapter"/> over embedding rows (<c>decoder.embedding_input</c>); <see cref="VibeVoiceRecognizer"/> runs it.
/// </summary>
/// <remarks>
/// Hyper-parameters: the language model's <c>decoder.*</c> keys; <c>vibevoice_asr.{audio_token_id,audio_bos_token_id,audio_eos_token_id,im_end_token_id,eos_token_id,sample_rate,hidden,normalize_dbfs}</c>;
/// the tokenizers' <c>vibevoice_asr.{acoustic,semantic}.{filters,kernel,depths,ratios,latent,eps}</c> (<see cref="VibeVoiceTokenizerParameters"/>); the prompt texts
/// <c>vibevoice_asr.prompt.{template,context_template}</c> with the placeholders <c>{audio}</c>, <c>{duration}</c> and <c>{context}</c>. Weights: the language model's canonical decoder names;
/// <c>{acoustic,semantic}.encoder.</c> (see <see cref="VibeVoiceAdapter"/>); <c>{acoustic,semantic}_connector.{fc1,norm,fc2}</c>.
/// </remarks>
internal sealed class VibeVoiceAsrAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "vibevoice_asr";

    /// <summary>The model family recorded in the language model component's configuration.</summary>
    public const string Family = "vibevoice_asr";

    /// <summary>The role of the language model component.</summary>
    public const string TextModelRole = "text-model";

    /// <summary>The role of the audio encoder component (graph <see cref="EncodeGraph"/>).</summary>
    public const string AudioEncoderRole = "audio-encoder";

    /// <summary>The audio encoder graph: <see cref="EncodeFrames"/> frames of audio [frames · hop, 1] to language model rows [frames, hidden].</summary>
    public const string EncodeGraph = "asr.encode";

    /// <summary>The frames one run of the audio encoder covers (a longer recording runs in several, the convolution caches carry on; a shorter tail is padded with silence).</summary>
    public const int EncodeFrames = 64;

    /// <inheritdoc/>
    public string Name => "VibeVoice ASR";

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
        var acoustic = VibeVoiceTokenizerParameters.From(src, a + "acoustic.");
        var semantic = VibeVoiceTokenizerParameters.From(src, a + "semantic.");
        var graphs = new Dictionary<string, ComputeGraph>(backbone.Graphs, StringComparer.Ordinal)
        {
            [EncodeGraph] = BuildEncode(acoustic, semantic, weights.GetValueOrDefault),
        };
        int hidden = src.Get<int>(a + "hidden");
        var config = new List<(string, object)>
        {
            ("family", Family), ("audioToken", src.Get<int>(a + "audio_token_id")), ("audioBos", src.Get<int>(a + "audio_bos_token_id")),
            ("audioEos", src.Get<int>(a + "audio_eos_token_id")), ("imEnd", src.Get<int>(a + "im_end_token_id")), ("eos", src.Get<int>(a + "eos_token_id")),
            ("hidden", hidden), ("sampleRate", src.Get<int>(a + "sample_rate")), ("hop", (int)acoustic.Hop), ("normalizeDbfs", src.Get<float>(a + "normalize_dbfs")),
            ("prompt.template", src.Get<string>(a + "prompt.template")), ("prompt.contextTemplate", src.Get<string>(a + "prompt.context_template")),
        };
        string[] ownPrefixes = ["acoustic.", "semantic.", "acoustic_connector.", "semantic_connector."];
        var backbonePrefixes = backbone.Weights.Keys.Where(k => !ownPrefixes.Any(p => k.StartsWith(p, StringComparison.Ordinal)))
            .Select(k => k[..(k.IndexOf('.', StringComparison.Ordinal) + 1)]).Where(p => p.Length > 0).Distinct().Order(StringComparer.Ordinal).ToList();
        var description = new ModelDescription
        {
            Kind = ModelKinds.SpeechRecognition,
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
                    Role = AudioEncoderRole,
                    Graphs = [EncodeGraph],
                    WeightPrefixes = ["acoustic.", "semantic.", "acoustic_connector.", "semantic_connector."],
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

    // Audio to language model rows: both tokenizers' encoders, then the two connectors summed.
    private static ComputeGraph BuildEncode(VibeVoiceTokenizerParameters acoustic, VibeVoiceTokenizerParameters semantic, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(EncodeGraph, weights);
        var audio = b.Input(AudioCodecGraphs.AudioInput, InputRole.Media, DataType.F32, new Shape(EncodeFrames * acoustic.Hop, 1));
        var acousticLatents = VibeVoiceTokenizerGraph.Encode(b, acoustic, audio, VibeVoiceAdapter.AcousticPrefix, VibeVoiceAdapter.AcousticPrefix);
        var semanticLatents = VibeVoiceTokenizerGraph.Encode(b, semantic, audio, VibeVoiceAdapter.SemanticPrefix, VibeVoiceAdapter.SemanticPrefix);
        b.Output("rows", b.Add(VibeVoiceAdapter.Connector(b, acousticLatents, "acoustic_connector"), VibeVoiceAdapter.Connector(b, semanticLatents, "semantic_connector")));
        return b.Build();
    }
}
