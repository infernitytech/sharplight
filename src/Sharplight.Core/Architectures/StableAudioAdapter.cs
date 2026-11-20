using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Architectures;

/// <summary>
/// Stable Audio Open (canonical arch <c>stable_audio</c>, kind <see cref="ModelKinds.AudioGeneration"/>): a T5 text encoder, a timing conditioner (start and total
/// seconds), a diffusion transformer trained with the v objective (sampled with DPM-Solver++(2M) SDE and classifier-free guidance) over the 21.5 Hz latents of a 44.1 kHz
/// stereo Oobleck autoencoder. Builds the graphs of <see cref="TimedAudioDiffusionGraphs"/> (see <see cref="StableAudioGraph"/>).
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>stable_audio.*</c>): see <see cref="StableAudioParameters"/> (the keys are the snake-case names of its members, the timing range under
/// <c>timing.</c>, the decoder's strides as <c>vae.upsampling_ratios</c>) and the T5 keys of <see cref="T5TextHyperParameters"/> under <c>text.</c> (with
/// <c>text.prompt_length</c>). Weights: <c>text.*</c> (the T5 encoder in the vocabulary of <see cref="EncoderDecoderAdapter"/>), <c>timing.{start, total}.{frequencies,
/// projection.weight, projection.bias}</c>, <c>dit.*</c> (<c>time_embedding.{frequencies, linear_1, linear_2}</c>, <c>global_embedding.{linear_1, linear_2}</c>,
/// <c>condition_embedding.{linear_1, linear_2}</c>, <c>input_mix</c> and <c>output_mix</c> (the 1×1 convolutions as [C, C] matrices), <c>proj_in</c>, <c>proj_out</c>,
/// <c>layers.{i}.{attention_norm, attention.{q,k,v,o}, cross_attention_norm, cross_attention.{q,k,v,o}, mlp_norm, mlp.up, mlp.down}</c>, the up projection's value half
/// before its gate half) and <c>vae.decoder.*</c> (see <see cref="OobleckDecoderGraph"/>).
/// </remarks>
public sealed class StableAudioAdapter : IModelAdapter
{
    /// <summary>The attachment that carries the tokenizer's JSON.</summary>
    public const string TokenizerAttachment = "tokenizer/json";

    private const int DecodeFrames = 256;
    private const int DecodeOverlap = 16;

    /// <inheritdoc/>
    public string Name => "Stable Audio audio generation";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [StableAudioParameters.Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var p = StableAudioParameters.From(src);
        var text = T5TextHyperParameters.Read(src, StableAudioParameters.Prefix, "text");
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
                    Config = ComponentConfig.Of((TimedAudioDiffusionGraphs.PromptLengthKey, text.ContextLength), (TimedAudioDiffusionGraphs.WidthKey, text.HiddenSize)),
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
                    WeightPrefixes = ["timing."],
                    Config = ComponentConfig.Of((TimedAudioDiffusionGraphs.WidthKey, p.ConditionWidth), (TimedAudioDiffusionGraphs.MaxSecondsKey, (double)p.MaxSeconds)),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Denoiser,
                    Graphs = [TimedAudioDiffusionGraphs.DenoiseGraph],
                    WeightPrefixes = ["dit."],
                    Config = ComponentConfig.Of((AudioGenerationGraphs.SampleRateKey, p.SampleRate), (AudioGenerationGraphs.ChannelsKey, p.AudioChannels),
                        (AudioGenerationGraphs.SamplesPerFrameKey, p.SamplesPerFrame), (AudioGenerationGraphs.LatentChannelsKey, p.LatentChannels),
                        (TimedAudioDiffusionGraphs.FramesKey, p.Frames), (AudioGenerationGraphs.DefaultStepsKey, p.DefaultSteps),
                        (TimedAudioDiffusionGraphs.DefaultGuidanceKey, (double)p.DefaultGuidance), (TimedAudioDiffusionGraphs.SigmaMinKey, (double)p.SigmaMin),
                        (TimedAudioDiffusionGraphs.SigmaMaxKey, (double)p.SigmaMax), (TimedAudioDiffusionGraphs.SigmaDataKey, (double)p.SigmaData)),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Vae,
                    Graphs = [AudioGenerationGraphs.DecodeGraph],
                    WeightPrefixes = ["vae."],
                    Config = ComponentConfig.Of((AudioGenerationGraphs.SampleRateKey, p.SampleRate), (AudioGenerationGraphs.ChannelsKey, p.AudioChannels),
                        (AudioGenerationGraphs.SamplesPerFrameKey, p.SamplesPerFrame), (AudioGenerationGraphs.DecodeFramesKey, DecodeFrames),
                        (AudioGenerationGraphs.DecodeOverlapKey, DecodeOverlap)),
                    SymbolBounds = new Dictionary<string, long> { [WellKnown.TokenSymbol] = DecodeFrames + (2 * DecodeOverlap) },
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
                [TimedAudioDiffusionGraphs.TextGraph] = T5TextGraphBuilder.Build(src.Weight, text, "text.", TimedAudioDiffusionGraphs.TextGraph, TimedAudioDiffusionGraphs.MaskInput),
                [TimedAudioDiffusionGraphs.TimingGraph] = StableAudioGraph.BuildTiming(p, src.Weight),
                [TimedAudioDiffusionGraphs.DenoiseGraph] = StableAudioGraph.BuildDenoiser(p, text.ContextLength, src.Weight),
                [AudioGenerationGraphs.DecodeGraph] = StableAudioGraph.BuildDecoder(p, src.Weight),
            },
            Attachments = attachments,
            SourceMetadata = metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
