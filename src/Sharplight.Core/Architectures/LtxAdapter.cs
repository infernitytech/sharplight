using System.Text;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.VideoGeneration;

namespace Sharplight.Architectures;

/// <summary>
/// Builds LTX-Video generators (architecture <c>ltx</c>: LTX-Video 0.9.0 text-to-video) as a <see cref="ModelKinds.VideoGeneration"/>
/// model: the T5 text encoder (unmasked, as the LTX pipeline runs it), the diffusion transformer with a key-masked cross-attention
/// (<see cref="LtxTransformerGraphBuilder"/>) and the non-causal VAE decoder over the whole latent video (<see cref="LtxVaeGraphBuilder"/>),
/// as the graphs of <see cref="VideoDiffusionGraphs"/>, plus the tokenizer and a flow-matching scheduler component (dynamic shift with the
/// terminal stretch). The built-in video diffusion pipeline runs it.
/// </summary>
/// <remarks>
/// Metadata: <c>ltx.transformer.*</c> (<see cref="LtxHyperParameters"/>), <c>ltx.vae.*</c> (<c>latent_channels</c>, <c>patch_size</c>,
/// <c>spatial_downscale</c>, <c>temporal_downscale</c>), <c>ltx.text_encoder.*</c> (the encoder-decoder keys of the T5 encoder plus
/// <c>prompt_length</c>), the tokenizer under the canonical <c>tokenizer.*</c> keys, <c>ltx.scheduler.*</c> and the defaults
/// <c>ltx.default_width</c>, <c>default_height</c>, <c>default_frames</c>, <c>default_steps</c>, <c>default_guidance</c>, <c>frames_per_second</c>.
/// </remarks>
public sealed class LtxAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "ltx";

    private const string Prefix = Architecture + ".";
    private const string TextEncoderPrefix = "text_encoder.";
    private const string SchedulerAttachment = "scheduler/source-config.json";

    /// <inheritdoc/>
    public string Name => "LTX-Video (video DiT + non-causal 3-D VAE + T5)";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures { get; } = [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        ArgumentNullException.ThrowIfNull(sources);
        ArgumentNullException.ThrowIfNull(options);
        var src = sources.Primary;
        var transformer = LtxHyperParameters.Read(src, Prefix);
        var t5 = T5TextHyperParameters.Read(src, Prefix, "text_encoder");
        int latentChannels = src.Get<int>(Prefix + "vae.latent_channels");
        int patch = src.Get(Prefix + "vae.patch_size", 4);
        int spatial = src.Get(Prefix + "vae.spatial_downscale", 32);
        int temporal = src.Get(Prefix + "vae.temporal_downscale", 8);
        float fps = (float)src.Get(Prefix + "frames_per_second", 25.0);
        long height = options.Dimensions.GetValueOrDefault(VideoDiffusionGraphs.Height, src.Get(Prefix + "default_height", 480));
        long width = options.Dimensions.GetValueOrDefault(VideoDiffusionGraphs.Width, src.Get(Prefix + "default_width", 704));
        int defaultFrames = src.Get(Prefix + "default_frames", 161);
        long latentFrames = options.Dimensions.GetValueOrDefault(VideoDiffusionGraphs.FramesDimension, (defaultFrames - 1) / temporal + 1);
        if (height <= 0 || width <= 0 || height % spatial != 0 || width % spatial != 0 || latentFrames <= 0)
        {
            throw new ArgumentException($"Video size {width}x{height} must be a positive multiple of {spatial} pixels for {src.Get(CanonicalKeys.Name, src.Architecture)}.");
        }

        long latentHeight = height / spatial;
        long latentWidth = width / spatial;
        var graphs = new Dictionary<string, Graph.ComputeGraph>
        {
            [VideoDiffusionGraphs.TextEncode] = T5TextGraphBuilder.Build(src.Weight, t5, TextEncoderPrefix, VideoDiffusionGraphs.TextEncode),
            [VideoDiffusionGraphs.Denoise] = LtxTransformerGraphBuilder.Build(src.Weight, transformer, latentFrames, latentHeight, latentWidth, t5.ContextLength,
                fps, temporal, spatial),
            [VideoDiffusionGraphs.VaeDecode] = LtxVaeGraphBuilder.BuildDecoder(src.Weight, latentChannels, patch, latentFrames, latentHeight, latentWidth,
                src.Get(Prefix + "vae.decode_timestep", 0f), src.Get(Prefix + "vae.upsample_residual", Array.Empty<bool>()),
                src.Get(Prefix + "vae.upsample_factor", Array.Empty<int>())),
        };
        var attachments = new Dictionary<string, IWeightSource>(StringComparer.Ordinal);
        if (src.Get<string?>(Prefix + "scheduler.source_config", null) is { } schedulerJson)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(schedulerJson);
            attachments[SchedulerAttachment] = WeightInfo.Derived(SchedulerAttachment, DataType.U8, new Shape(bytes.Length), () => bytes).Source;
        }

        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = graphs,
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Attachments = attachments,
            Description = Describe(src, transformer, t5, latentChannels, spatial, temporal, attachments.Count > 0),
        };
    }

    private static ModelDescription Describe(ModelSource src, LtxHyperParameters transformer, T5TextHyperParameters t5, int latentChannels, int spatial,
        int temporal, bool hasSchedulerAttachment)
    {
        string s = Prefix + "scheduler.";
        var scheduler = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.TrainSteps, src.Get(s + "train_steps", 1000))
            .Add(DiffusionConfigKeys.PredictionType, "flow")
            .Add(DiffusionConfigKeys.DynamicShift, src.Get(s + "dynamic_shift", true))
            .Add(DiffusionConfigKeys.Shift, src.Get(s + "shift", 1.0))
            .Add(DiffusionConfigKeys.BaseShift, src.Get(s + "base_shift", 0.95))
            .Add(DiffusionConfigKeys.MaxShift, src.Get(s + "max_shift", 2.05))
            .Add(DiffusionConfigKeys.BaseSequenceLength, src.Get(s + "base_sequence_length", 1024))
            .Add(DiffusionConfigKeys.MaxSequenceLength, src.Get(s + "max_sequence_length", 4096))
            .Add(DiffusionConfigKeys.ShiftTerminal, src.Get(s + "shift_terminal", 0.1))
            .Add(DiffusionConfigKeys.Sampler, src.Get(s + "sampler", Samplers.Euler));
        return new ModelDescription
        {
            Kind = ModelKinds.VideoGeneration,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.TextEncoder,
                    Graphs = [VideoDiffusionGraphs.TextEncode],
                    WeightPrefixes = [TextEncoderPrefix],
                    Config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, t5.ContextLength).Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, t5.ContextLength).Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Transformer,
                    Graphs = [VideoDiffusionGraphs.Denoise],
                    WeightPrefixes = ["transformer."],
                    Config = new ComponentConfigBuilder()
                        .Add(DiffusionConfigKeys.LatentChannels, transformer.LatentChannels)
                        .Add(DiffusionConfigKeys.PatchSize, 1)
                        .Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Vae,
                    Graphs = [VideoDiffusionGraphs.VaeDecode],
                    WeightPrefixes = ["vae."],
                    Config = new ComponentConfigBuilder()
                        .Add(DiffusionConfigKeys.LatentChannels, latentChannels)
                        .Add(DiffusionConfigKeys.Downscale, spatial)
                        .Add(VideoConfigKeys.TemporalDownscale, temporal)
                        .Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Scheduler,
                    Config = scheduler.Build(),
                    Attachments = hasSchedulerAttachment
                        ? new Dictionary<string, string> { ["source-config"] = SchedulerAttachment }
                        : new Dictionary<string, string>(),
                },
            ],
            Config = new ComponentConfigBuilder()
                .Add(DiffusionConfigKeys.Width, src.Get(Prefix + "default_width", 704))
                .Add(DiffusionConfigKeys.Height, src.Get(Prefix + "default_height", 480))
                .Add(VideoConfigKeys.Frames, src.Get(Prefix + "default_frames", 161))
                .Add(VideoConfigKeys.FramesPerSecond, src.Get(Prefix + "frames_per_second", 25.0))
                .Add(DiffusionConfigKeys.Steps, src.Get(Prefix + "default_steps", 50))
                .Add(DiffusionConfigKeys.GuidanceScale, src.Get(Prefix + "default_guidance", 3.0))
                .Build(),
        };
    }
}
