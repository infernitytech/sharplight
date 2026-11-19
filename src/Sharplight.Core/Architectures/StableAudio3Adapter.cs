using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Tokenizers;

namespace Sharplight.Architectures;

/// <summary>
/// Stable Audio 3 (canonical arch <c>stable_audio_3</c>, kind <see cref="ModelKinds.AudioGeneration"/>): a T5Gemma text encoder, a seconds conditioner, a diffusion
/// transformer with adaLN modulation and memory rows trained as a rectified flow (the base models sampled with Euler steps and adaptive projected guidance, the
/// post-trained ones with ping-pong steps) over the 10.8 Hz, 256-channel latents of the SAME transformer autoencoder (44.1 kHz stereo). Builds the rectified-flow form of
/// <see cref="TimedAudioDiffusionGraphs"/> (see <see cref="StableAudio3Graph"/>, <see cref="T5GemmaEncoderGraph"/> and <see cref="SameDecoderGraph"/>).
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>stable_audio_3.*</c>): see <see cref="StableAudio3Parameters"/> (snake-case keys; the seconds range under <c>seconds.</c>, the sampling schedule
/// under <c>schedule.</c>: <c>type</c>, <c>rate</c>, <c>anchor_logsnr</c>, <c>anchor_length</c>, <c>logsnr_end</c>, <c>min_length</c>, <c>max_length</c>,
/// <c>alpha_min</c>, <c>alpha_max</c>, <c>base_shift</c>, <c>max_shift</c>), the T5Gemma keys of <see cref="T5GemmaEncoderParameters"/> under <c>text.</c> and the decoder
/// keys of <see cref="SameDecoderParameters"/> under <c>vae.</c>. Weights: <c>text.*</c> (<see cref="T5GemmaEncoderGraph"/>, with <c>text.padding</c>),
/// <c>seconds.projection.{weight, bias}</c>, <c>dit.*</c> (<c>time_embedding</c>, <c>condition_embedding</c>, <c>global_embedding</c> and <c>modulation_embedding</c> as
/// <c>linear_1</c> and <c>linear_2</c>, <c>input_mix</c> and <c>output_mix</c> (the 1×1 convolutions as [C, C] matrices), <c>proj_in</c>, <c>proj_out</c>,
/// <c>memory_tokens</c>, <c>layers.{i}.{modulation, attention_norm, attention.{qkv, q_norm, k_norm, o}, cross_attention_norm, cross_attention.{q, kv, q_norm, k_norm, o},
/// local_embedding.{linear_1, linear_2}, mlp_norm, mlp.up, mlp.down}</c>) and <c>vae.*</c> (<see cref="SameDecoderGraph"/>).
/// </remarks>
public sealed class StableAudio3Adapter : IModelAdapter
{
    /// <summary>The attachment that carries the tokenizer's JSON.</summary>
    public const string TokenizerAttachment = "tokenizer/json";

    /// <inheritdoc/>
    public string Name => "Stable Audio 3 audio generation";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [StableAudio3Parameters.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = StableAudio3Parameters.From(src);
        var text = T5GemmaEncoderParameters.From(src, "text.");
        var vae = SameDecoderParameters.From(src, "vae.");
        var tokenizer = TokenizerData.FromMetadata(src.Metadata);
        var attachments = new Dictionary<string, IWeightSource>(src.Attachments, StringComparer.Ordinal)
        {
            [TokenizerAttachment] = new MemoryWeightSource(tokenizer.ToJson()),
        };
        var metadata = src.Metadata.Where(e => !e.Key.StartsWith("tokenizer.", StringComparison.Ordinal)).ToDictionary(StringComparer.Ordinal);
        var description = new ModelDescription
        {
            Kind = ModelKinds.AudioGeneration,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.TextEncoder,
                    Graphs = [TimedAudioDiffusionGraphs.TextGraph],
                    WeightPrefixes = ["text."],
                    Config = ComponentConfig.Of((TimedAudioDiffusionGraphs.PromptLengthKey, p.PromptLength), (TimedAudioDiffusionGraphs.WidthKey, text.Hidden)),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Attachments = new Dictionary<string, string>(StringComparer.Ordinal) { [ModelComponentAttachments.Tokenizer] = TokenizerAttachment },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Encoder,
                    Graphs = [TimedAudioDiffusionGraphs.TimingGraph],
                    WeightPrefixes = ["seconds."],
                    Config = ComponentConfig.Of((TimedAudioDiffusionGraphs.WidthKey, p.ConditionWidth), (TimedAudioDiffusionGraphs.MaxSecondsKey, (double)p.MaxSeconds)),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Denoiser,
                    Graphs = [TimedAudioDiffusionGraphs.DenoiseGraph],
                    WeightPrefixes = ["dit."],
                    Config = DenoiserConfig(src, p),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = p.MaxFrames },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Vae,
                    Graphs = [AudioGenerationGraphs.DecodeGraph],
                    WeightPrefixes = ["vae."],
                    Config = ComponentConfig.Of((AudioGenerationGraphs.SampleRateKey, p.SampleRate), (AudioGenerationGraphs.ChannelsKey, p.AudioChannels),
                        (AudioGenerationGraphs.SamplesPerFrameKey, p.SamplesPerFrame), (AudioGenerationGraphs.DecodeFramesKey, vae.DecodeFrames),
                        (AudioGenerationGraphs.DecodeOverlapKey, vae.DecodeOverlap), (TimedAudioDiffusionGraphs.NoiseRowsKey, vae.Stride),
                        (TimedAudioDiffusionGraphs.NoiseWidthKey, vae.Width), (TimedAudioDiffusionGraphs.LatentNoiseKey, (double)vae.LatentNoise)),
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph>(StringComparer.Ordinal)
            {
                [TimedAudioDiffusionGraphs.TextGraph] = T5GemmaEncoderGraph.Build(TimedAudioDiffusionGraphs.TextGraph, src.Weight, text, "text.", p.PromptLength,
                    TimedAudioDiffusionGraphs.TokensInput, TimedAudioDiffusionGraphs.MaskInput, TimedAudioDiffusionGraphs.HiddenOutput, "text.padding"),
                [TimedAudioDiffusionGraphs.TimingGraph] = StableAudio3Graph.BuildSeconds(p, src.Weight),
                [TimedAudioDiffusionGraphs.DenoiseGraph] = StableAudio3Graph.BuildDenoiser(p, src.Weight),
                [AudioGenerationGraphs.DecodeGraph] = BuildDecoder(p, vae, src.Weight),
            },
            Attachments = attachments,
            SourceMetadata = metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }

    private static IReadOnlyDictionary<string, System.Text.Json.JsonElement> DenoiserConfig(ModelSource src, StableAudio3Parameters p)
    {
        var entries = new List<(string, object)>
        {
            (AudioGenerationGraphs.SampleRateKey, p.SampleRate),
            (AudioGenerationGraphs.ChannelsKey, p.AudioChannels),
            (AudioGenerationGraphs.SamplesPerFrameKey, p.SamplesPerFrame),
            (AudioGenerationGraphs.LatentChannelsKey, p.LatentChannels),
            (TimedAudioDiffusionGraphs.FramesKey, p.MaxFrames),
            (AudioGenerationGraphs.DefaultStepsKey, p.DefaultSteps),
            (TimedAudioDiffusionGraphs.DefaultGuidanceKey, (double)p.DefaultGuidance),
            (TimedAudioDiffusionGraphs.ObjectiveKey, p.Objective),
            (TimedAudioDiffusionGraphs.MemoryTokensKey, p.MemoryTokens),
            (TimedAudioDiffusionGraphs.FrameAlignKey, p.FrameAlign),
            (TimedAudioDiffusionGraphs.HeadroomSecondsKey, (double)p.HeadroomSeconds),
            (TimedAudioDiffusionGraphs.TimeLogSnrKey, p.TimeLogSnr),
            (TimedAudioDiffusionGraphs.LocalChannelsKey, p.LocalChannels),
            (TimedAudioDiffusionGraphs.ScheduleKey, src.Arch("schedule.type", "logsnr")),
        };
        foreach (var (key, name) in new[]
        {
            (TimedAudioDiffusionGraphs.AnchorLogSnrKey, "anchor_logsnr"), (TimedAudioDiffusionGraphs.LogSnrEndKey, "logsnr_end"), (TimedAudioDiffusionGraphs.ShiftRateKey, "rate"),
            (TimedAudioDiffusionGraphs.AnchorLengthKey, "anchor_length"), (TimedAudioDiffusionGraphs.MinLengthKey, "min_length"), (TimedAudioDiffusionGraphs.MaxLengthKey, "max_length"),
            (TimedAudioDiffusionGraphs.AlphaMinKey, "alpha_min"), (TimedAudioDiffusionGraphs.AlphaMaxKey, "alpha_max"), (TimedAudioDiffusionGraphs.BaseShiftKey, "base_shift"),
            (TimedAudioDiffusionGraphs.MaxShiftKey, "max_shift"),
        })
        {
            if (src.Metadata.ContainsKey(StableAudio3Parameters.Prefix + "schedule." + name))
            {
                entries.Add((key, (double)src.Arch<float>("schedule." + name)));
            }
        }

        return ComponentConfig.Of([.. entries]);
    }

    private static ComputeGraph BuildDecoder(StableAudio3Parameters p, SameDecoderParameters vae, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioGenerationGraphs.DecodeGraph, weights);
        var latent = b.Input(AudioGenerationGraphs.LatentInput, InputRole.Other, DataType.F32, new Shape(vae.DecodeFrames, p.LatentChannels));
        var noise = b.Input(TimedAudioDiffusionGraphs.NoiseInput, InputRole.Other, DataType.F32, new Shape((long)vae.DecodeFrames * vae.Stride, vae.Width));
        b.Output(AudioGenerationGraphs.AudioOutput, SameDecoderGraph.Build(b, latent, noise, vae, "vae.", vae.DecodeFrames, p.AudioChannels));
        return b.Build();
    }
}
