using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Dia (canonical architecture <see cref="Architecture"/>; Nari Labs' dialogue text-to-speech model) with its audio codec: a composite adapter (<see cref="ICompositeModelAdapter"/>)
/// whose primary source is the encoder-decoder transformer and whose <see cref="AuxiliaryRoles.Vocoder"/> source is the 44.1 kHz Descript Audio Codec (<see cref="DacAdapter"/>'s
/// architecture, weights under <c>codec.</c>). The encoder reads the text as bytes (<c>[S1]</c> and <c>[S2]</c> are the bytes 1 and 2); the decoder predicts the nine codebooks of
/// every frame at once, each delayed by its own number of frames (the delay pattern), with classifier-free guidance run by the synthesizer. The codec's encoder turns an audio prompt
/// into codes, its decoder the generated codes into audio. <see cref="DiaSynthesizer"/> runs it.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>dia.*</c>): <c>layer_count</c>, <c>hidden_size</c>, <c>attention.head_count</c>, <c>attention.head_count_kv</c>, <c>head_dim</c>, <c>cross_attention.head_count</c>,
/// <c>cross_attention.head_dim</c>, <c>intermediate_size</c>, <c>channels</c>, <c>vocabulary</c>, <c>context_length</c>, <c>norm_eps</c>, <c>rope.base</c>, <c>delay_pattern</c> (long[]),
/// <c>bos_token</c>, <c>eos_token</c>, <c>pad_token</c>, <c>encoder.{layer_count,hidden_size,attention.head_count,head_dim,intermediate_size,vocabulary,context_length}</c> and the
/// sampling defaults <c>sampling.{guidance_scale,temperature,top_k,top_p,guidance_top_k}</c>. Weights: <c>encoder.token_embedding.weight</c> [vocabulary, width],
/// <c>encoder.layers.{i}.{attention_norm,mlp_norm}.weight</c>, <c>attention.{q,k,v,o}.weight</c> and <c>mlp.{gate,up,down}.weight</c> as [out, in], <c>encoder.output_norm.weight</c>; the
/// decoder the same under <c>decoder.</c> plus <c>cross_attention_norm.weight</c> and <c>cross_attention.{q,k,v,o}.weight</c> per layer, <c>decoder.embedding.weight</c>
/// [channels · vocabulary, hidden] (codebook c's rows start at c · vocabulary), <c>decoder.output_norm.weight</c> and <c>decoder.lm_head.weight</c> [channels · vocabulary, hidden];
/// the codec under <c>codec.</c>.
/// </remarks>
internal sealed class DiaAdapter : ICompositeModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = DiaParameters.Architecture;

    /// <summary>The model family recorded in the decoder component's configuration.</summary>
    public const string Family = "dia";

    /// <summary>The most rows one decoder pass takes (longer audio prompts run in several passes).</summary>
    public const int MaxRows = 256;

    /// <summary>The most frames one codec decode covers.</summary>
    public const int CodecDecodeFrames = 1100;

    /// <summary>The frames before each codec decode chunk the decoder sees as context (their audio is dropped).</summary>
    public const int CodecContextFrames = 32;

    /// <summary>The most frames of an audio prompt.</summary>
    public const int PromptFrames = 860;

    /// <inheritdoc/>
    public string Name => "Dia";

    /// <inheritdoc/>
    public bool CanLoad(ModelSourceSet sources)
        => sources.Primary.Architecture == Architecture && sources.Auxiliary.TryGetValue(AuxiliaryRoles.Vocoder, out var codec) && codec.Architecture == DacParameters.Architecture;

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options, ArchitectureCatalog architectures)
    {
        var src = sources.Primary;
        var codecSource = sources.Auxiliary[AuxiliaryRoles.Vocoder];
        var p = DiaParameters.From(src);
        var codec = DacParameters.From(codecSource);
        if (codec.Codebooks != p.Channels)
        {
            throw new NotSupportedException($"{Architecture}: the model predicts {p.Channels} codebooks, the codec has {codec.Codebooks}.");
        }

        var weights = new Dictionary<string, WeightInfo>(src.Weights, StringComparer.Ordinal);
        foreach (var (name, weight) in codecSource.Weights)
        {
            weights[DiaGraphs.CodecPrefix + name] = weight.As(DiaGraphs.CodecPrefix + name);
        }

        WeightInfo? Lookup(string n) => weights.GetValueOrDefault(DiaGraphs.CodecPrefix + n);
        var states = DiaGraph.States(p, options);
        var graphs = new Dictionary<string, ComputeGraph>(StringComparer.Ordinal)
        {
            [DiaGraphs.TextGraph] = DiaGraph.BuildText(p, weights.GetValueOrDefault),
            [DiaGraphs.StepGraph] = DiaGraph.BuildStep(p, weights.GetValueOrDefault, states),
            [AudioCodecGraphs.EncodeGraph] = DacGraph.BuildEncoder(codec, Lookup),
            [AudioCodecGraphs.DecodeGraph] = DacGraph.BuildDecoder(codec, Lookup),
        };
        var config = new List<(string, object)>
        {
            ("family", Family), ("channels", p.Channels), ("vocabulary", p.Vocabulary), ("bos", p.BosToken), ("eos", p.EosToken), ("pad", p.PadToken), ("maxText", p.MaxText),
            ("delays", string.Join(',', p.DelayPattern)), ("frameRate", (double)codec.SampleRate / codec.SamplesPerFrame), ("codecSize", codec.CodebookSize),
        };
        foreach (var key in new[] { "guidance_scale", "temperature", "top_k", "top_p", "guidance_top_k" })
        {
            if (src.Has("dia.sampling." + key))
            {
                config.Add(("sampling." + key, src.Get<float>("dia.sampling." + key)));
            }
        }

        var codecConfig = ComponentConfig.Of((AudioCodecGraphs.SampleRateKey, codec.SampleRate), (AudioCodecGraphs.SamplesPerFrameKey, (int)codec.SamplesPerFrame),
            (AudioCodecGraphs.CodebooksKey, codec.Codebooks), (AudioCodecGraphs.DefaultCodebooksKey, codec.Codebooks), (AudioCodecGraphs.CodebookSizeKey, codec.CodebookSize));
        var description = new ModelDescription
        {
            Kind = ModelKinds.SpeechSynthesis,
            Components =
            [
                new ModelComponent
                {
                    Role = DiaGraphs.TextRole,
                    Graphs = [DiaGraphs.TextGraph],
                    WeightPrefixes = ["encoder."],
                    Config = ComponentConfig.Of(("width", p.EncoderHidden)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = p.MaxText },
                },
                new ModelComponent
                {
                    Role = DiaGraphs.DecoderRole,
                    Graphs = [DiaGraphs.StepGraph],
                    WeightPrefixes = ["decoder."],
                    Config = ComponentConfig.Of([.. config]),
                    SymbolBounds = new Dictionary<string, long>
                    {
                        [WellKnown.TokenSymbol] = MaxRows,
                        [WellKnown.ContextSymbol] = p.MaxAudio + p.MaxDelay + 1,
                        [WellKnown.EncoderSymbol] = p.MaxText,
                    },
                },
                new ModelComponent
                {
                    Role = DiaGraphs.CodecEncoderRole,
                    Graphs = [AudioCodecGraphs.EncodeGraph],
                    WeightPrefixes = [DiaGraphs.CodecPrefix],
                    Config = codecConfig,
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = PromptFrames * codec.SamplesPerFrame },
                },
                new ModelComponent
                {
                    Role = DiaGraphs.CodecDecoderRole,
                    Graphs = [AudioCodecGraphs.DecodeGraph],
                    WeightPrefixes = [DiaGraphs.CodecPrefix],
                    Config = codecConfig,
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = CodecDecodeFrames },
                },
            ],
            Config = ComponentConfig.Of(("sampleRate", codec.SampleRate)),
        };
        return new ModelDefinition
        {
            Architecture = Architecture,
            Name = src.Get(CanonicalKeys.Name, Architecture),
            Weights = weights,
            Graphs = graphs,
            States = [.. states],
            SourceMetadata = src.Metadata,
            Description = description,
            Attachments = src.Attachments,
            OwnedResources = [.. src.OwnedResources, .. codecSource.OwnedResources],
        };
    }
}
