using System.Text;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.VideoGeneration;

namespace Sharplight.Architectures;

/// <summary>
/// Builds Wan-style video generators (architecture <c>wan</c>: Wan 2.1 text-to-video, Wan 2.2 TI2V text- and image-to-video) as a
/// <see cref="ModelKinds.VideoGeneration"/> model: the umT5 text encoder with a key mask (<see cref="T5TextGraphBuilder"/>), the video
/// diffusion transformer (<see cref="WanTransformerGraphBuilder"/>) and the causal 3-D VAE streamed over latent frames
/// (<see cref="WanVaeGraphBuilder"/>), as the graphs of <see cref="VideoDiffusionGraphs"/>, plus the tokenizer and a flow-matching
/// scheduler component. The built-in video diffusion pipeline runs it.
/// </summary>
/// <remarks>
/// Canonical vocabulary: the transformer as documented on <see cref="WanTransformerGraphBuilder"/>, the VAE as on
/// <see cref="WanVaeGraphBuilder"/>, the umT5 encoder under <c>text_encoder.</c> in the encoder vocabulary of
/// <see cref="EncoderDecoderAdapter"/>. Metadata: <c>wan.transformer.*</c> (<see cref="WanHyperParameters"/>), <c>wan.vae.*</c>
/// (<see cref="WanVaeHyperParameters"/>), <c>wan.text_encoder.*</c> (the encoder-decoder keys of the umT5 encoder plus
/// <c>prompt_length</c>), the tokenizer under the canonical <c>tokenizer.*</c> keys, <c>wan.scheduler.*</c> (<c>train_steps</c>,
/// <c>shift</c>, <c>sampler</c>, <c>source_config</c>), and the defaults <c>wan.default_width</c>, <c>default_height</c>,
/// <c>default_frames</c>, <c>default_steps</c>, <c>default_guidance</c>, <c>frames_per_second</c>, <c>first_frame_conditioning</c>
/// and <c>negative_prompt</c>.
/// </remarks>
public sealed class WanAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "wan";

    private const string Prefix = Architecture + ".";
    private const string TextEncoderPrefix = "text_encoder.";
    private const string TransformerPrefix = "transformer.";
    private const string SecondTransformerPrefix = "transformer_2.";
    private const string VaePrefix = "vae.";
    private const string SchedulerAttachment = "scheduler/source-config.json";

    /// <inheritdoc/>
    public string Name => "Wan (video DiT + causal 3-D VAE + umT5)";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures { get; } = [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        ArgumentNullException.ThrowIfNull(sources);
        ArgumentNullException.ThrowIfNull(options);
        var src = sources.Primary;
        RequireComponents(src);
        var transformer = WanHyperParameters.Read(src, Prefix);
        var vae = WanVaeHyperParameters.Read(src, Prefix);
        var t5 = T5TextHyperParameters.Read(src, Prefix, "text_encoder");
        bool firstFrame = src.Get(Prefix + "first_frame_conditioning", false);
        int multiple = vae.SpatialDownscale * transformer.PatchSize[1];
        long height = options.Dimensions.GetValueOrDefault(VideoDiffusionGraphs.Height, src.Get(Prefix + "default_height", 480));
        long width = options.Dimensions.GetValueOrDefault(VideoDiffusionGraphs.Width, src.Get(Prefix + "default_width", 832));
        int defaultFrames = src.Get(Prefix + "default_frames", 81);
        long latentFrames = options.Dimensions.GetValueOrDefault(VideoDiffusionGraphs.FramesDimension, (defaultFrames - 1) / vae.TemporalDownscale + 1);
        if (height <= 0 || width <= 0 || height % multiple != 0 || width % multiple != 0 || latentFrames <= 0)
        {
            throw new ArgumentException($"Video size {width}x{height} must be a positive multiple of {multiple} pixels for {src.Get(CanonicalKeys.Name, src.Architecture)}.");
        }

        long latentHeight = height / vae.SpatialDownscale;
        long latentWidth = width / vae.SpatialDownscale;
        var graphs = new Dictionary<string, Graph.ComputeGraph>
        {
            [VideoDiffusionGraphs.TextEncode] = T5TextGraphBuilder.Build(src.Weight, t5, TextEncoderPrefix, VideoDiffusionGraphs.TextEncode, VideoDiffusionGraphs.TokenMask),
            [VideoDiffusionGraphs.Denoise] = WanTransformerGraphBuilder.Build(src.Weight, transformer, latentFrames, latentHeight, latentWidth, t5.ContextLength),
            [VideoDiffusionGraphs.VaeDecodeFirst] = WanVaeGraphBuilder.BuildDecoder(src.Weight, vae, latentHeight, latentWidth, first: true),
            [VideoDiffusionGraphs.VaeDecodeNext] = WanVaeGraphBuilder.BuildDecoder(src.Weight, vae, latentHeight, latentWidth, first: false),
        };
        bool twoExperts = src.Weight(SecondTransformerPrefix + "patch_embedding.weight") is not null;
        if (twoExperts)
        {
            graphs[VideoDiffusionGraphs.Denoise2] = WanTransformerGraphBuilder.Build(src.Weight, transformer, latentFrames, latentHeight, latentWidth, t5.ContextLength,
                SecondTransformerPrefix, VideoDiffusionGraphs.Denoise2);
        }

        bool hasEncoder = src.Weight(VaePrefix + "encoder.conv_in.weight") is not null;
        if (hasEncoder)
        {
            graphs[VideoDiffusionGraphs.VaeEncode] = WanVaeGraphBuilder.BuildEncoder(src.Weight, vae, height, width);
            if (!vae.Residual)
            {
                graphs[VideoDiffusionGraphs.VaeEncodeNext] = WanVaeGraphBuilder.BuildEncoderNext(src.Weight, vae, height, width);
            }
        }

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
            Description = Describe(src, transformer, vae, t5, hasEncoder, firstFrame && hasEncoder && transformer.TokenTimesteps, attachments.Count > 0, twoExperts),
        };
    }

    /// <summary>Fails with the list of missing parts when a transformer-only source comes without its companions.</summary>
    private static void RequireComponents(ModelSource src)
    {
        var missing = new List<string>();
        if (!src.Has(Prefix + "text_encoder.hidden_size") || !src.Has("tokenizer.model"))
        {
            missing.Add("text-encoder (umT5-XXL with its tokenizer)");
        }

        if (!src.Has(Prefix + "vae.latent_channels"))
        {
            missing.Add("vae");
        }

        if (missing.Count > 0)
        {
            throw new NotSupportedException($"'{src.Path}' holds only the Wan transformer; it needs its companions, given as auxiliary sources by role "
                + $"(a Wan diffusers folder of the same family serves as both): {string.Join(", ", missing)}.");
        }
    }

    private static ModelDescription Describe(ModelSource src, WanHyperParameters transformer, WanVaeHyperParameters vae, T5TextHyperParameters t5,
        bool hasEncoder, bool firstFrame, bool hasSchedulerAttachment, bool twoExperts)
    {
        string s = Prefix + "scheduler.";
        var scheduler = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.TrainSteps, src.Get(s + "train_steps", 1000))
            .Add(DiffusionConfigKeys.PredictionType, "flow")
            .Add(DiffusionConfigKeys.Shift, src.Get(s + "shift", 5f))
            .Add(DiffusionConfigKeys.TimestepSpacing, FlowMatchEulerSampler.SchedulerSpacing)
            .Add(DiffusionConfigKeys.Sampler, src.Get(s + "sampler", Samplers.UniPc));
        var vaeGraphs = new List<string> { VideoDiffusionGraphs.VaeDecodeFirst, VideoDiffusionGraphs.VaeDecodeNext };
        if (hasEncoder)
        {
            vaeGraphs.Add(VideoDiffusionGraphs.VaeEncode);
            if (!vae.Residual)
            {
                vaeGraphs.Add(VideoDiffusionGraphs.VaeEncodeNext);
            }
        }

        var model = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.Width, src.Get(Prefix + "default_width", 832))
            .Add(DiffusionConfigKeys.Height, src.Get(Prefix + "default_height", 480))
            .Add(VideoConfigKeys.Frames, src.Get(Prefix + "default_frames", 81))
            .Add(VideoConfigKeys.FramesPerSecond, src.Get(Prefix + "frames_per_second", 16.0))
            .Add(DiffusionConfigKeys.Steps, src.Get(Prefix + "default_steps", 50))
            .Add(DiffusionConfigKeys.GuidanceScale, src.Get(Prefix + "default_guidance", 5.0))
            .Add(VideoConfigKeys.FirstFrameConditioning, firstFrame);
        if (twoExperts)
        {
            model.Add(VideoConfigKeys.BoundaryRatio, src.Get(Prefix + "boundary_ratio", 0.875))
                .Add(VideoConfigKeys.GuidanceScale2, src.Get(Prefix + "default_guidance_2", src.Get(Prefix + "default_guidance", 5.0)));
        }

        if (src.Get<string?>(Prefix + "negative_prompt", null) is { } negative)
        {
            model.Add(VideoConfigKeys.NegativePrompt, negative);
        }

        var transformerConfig = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.LatentChannels, transformer.InChannels)
            .Add(DiffusionConfigKeys.PatchSize, transformer.PatchSize[1])
            .Add(VideoConfigKeys.TokenTimesteps, transformer.TokenTimesteps)
            .Build();
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
                    WeightPrefixes = [TransformerPrefix],
                    Config = transformerConfig,
                },
                .. (twoExperts
                    ?
                    [
                        new ModelComponent
                        {
                            Role = VideoDiffusionGraphs.SecondTransformerRole,
                            Graphs = [VideoDiffusionGraphs.Denoise2],
                            WeightPrefixes = [SecondTransformerPrefix],
                            Config = transformerConfig,
                        },
                    ]
                    : Array.Empty<ModelComponent>()),
                new ModelComponent
                {
                    Role = ModelComponentRoles.Vae,
                    Graphs = vaeGraphs,
                    WeightPrefixes = [VaePrefix],
                    Config = new ComponentConfigBuilder()
                        .Add(DiffusionConfigKeys.LatentChannels, vae.LatentChannels)
                        .Add(DiffusionConfigKeys.Downscale, vae.SpatialDownscale)
                        .Add(VideoConfigKeys.TemporalDownscale, vae.TemporalDownscale)
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
            Config = model.Build(),
        };
    }
}
