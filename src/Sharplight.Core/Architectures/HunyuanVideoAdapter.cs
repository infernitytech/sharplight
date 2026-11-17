using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.VideoGeneration;

namespace Sharplight.Architectures;

/// <summary>
/// Builds HunyuanVideo-style video generators (architecture <c>hunyuan_video</c>) as a <see cref="ModelKinds.VideoGeneration"/> model: a
/// decoder language model (LLaVA-Llama-3-8B) whose hidden states two layers before the last condition the denoiser
/// (<see cref="DecoderGraphBuilder"/> over the templated prompt), CLIP-L's pooled embedding (<see cref="ClipTextGraphBuilder"/>), the
/// dual- and single-stream video transformer (<see cref="HunyuanVideoTransformerGraphBuilder"/>) with the guidance scale embedded, and the
/// causal 3-D VAE decoder (<see cref="HunyuanVideoVaeGraphBuilder"/>), decoded in overlapping temporal tiles; plus both tokenizers and a
/// flow-matching Euler scheduler. The built-in video diffusion pipeline runs it.
/// </summary>
/// <remarks>
/// Metadata: <c>hunyuan_video.transformer.*</c> (<see cref="HunyuanVideoHyperParameters"/>), <c>hunyuan_video.vae.*</c>
/// (<c>latent_channels</c>, <c>scaling_factor</c>, <c>groups</c>, <c>temporal_compression</c>, <c>spatial_compression</c>,
/// <c>tile_frames</c>, <c>tile_stride</c>, <c>spatial_tile</c>, <c>spatial_tile_stride</c>), <c>hunyuan_video.text_encoder_source.*</c> (the language model as a canonical source of its own,
/// weights under <c>text_encoder.</c>), <c>hunyuan_video.text_encoder.*</c> (<c>prompt_length</c>, <c>prompt_template</c>,
/// <c>prompt_prefix</c> = the template tokens cropped from the states, <c>layer</c>), <c>hunyuan_video.text_encoder_2.*</c> (CLIP, weights
/// under <c>text_encoder_2.</c>) with its tokenizer under <c>hunyuan_video.tokenizer_2.</c>, <c>hunyuan_video.scheduler.*</c> and the
/// defaults <c>default_width</c>, <c>default_height</c>, <c>default_frames</c>, <c>default_steps</c>, <c>default_guidance</c>,
/// <c>frames_per_second</c>.
/// </remarks>
public sealed class HunyuanVideoAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "hunyuan_video";

    private const string Prefix = Architecture + ".";
    private const string TextEncoderSourcePrefix = Prefix + "text_encoder_source.";
    private const string TextEncoderPrefix = "text_encoder.";
    private const string SecondTextEncoderPrefix = "text_encoder_2.";
    private const string TransformerPrefix = "transformer.";
    private const string VaePrefix = "vae.";

    /// <inheritdoc/>
    public string Name => "HunyuanVideo (dual/single-stream video DiT + causal 3-D VAE + Llama + CLIP)";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures { get; } = [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        ArgumentNullException.ThrowIfNull(sources);
        ArgumentNullException.ThrowIfNull(options);
        var src = sources.Primary;
        RequireComponents(src);
        var transformer = HunyuanVideoHyperParameters.Read(src, Prefix);
        string v = Prefix + "vae.";
        int latentChannels = src.Get(v + "latent_channels", 16);
        int spatial = src.Get(v + "spatial_compression", 8);
        int temporal = src.Get(v + "temporal_compression", 4);
        int multiple = spatial * transformer.PatchSize;
        long height = options.Dimensions.GetValueOrDefault(VideoDiffusionGraphs.Height, src.Get(Prefix + "default_height", 720));
        long width = options.Dimensions.GetValueOrDefault(VideoDiffusionGraphs.Width, src.Get(Prefix + "default_width", 1280));
        int defaultFrames = src.Get(Prefix + "default_frames", 129);
        long latentFrames = options.Dimensions.GetValueOrDefault(VideoDiffusionGraphs.FramesDimension, (defaultFrames - 1) / temporal + 1);
        if (height <= 0 || width <= 0 || height % multiple != 0 || width % multiple != 0 || latentFrames <= 0)
        {
            throw new ArgumentException($"Video size {width}x{height} must be a positive multiple of {multiple} pixels for {src.Get(CanonicalKeys.Name, src.Architecture)}.");
        }

        long latentHeight = height / spatial;
        long latentWidth = width / spatial;
        int promptLength = src.Get(Prefix + "text_encoder.prompt_length", 256);
        int promptPrefix = src.Get(Prefix + "text_encoder.prompt_prefix", 0);
        var textSource = DecoderTextEncoder.Source(src, TextEncoderSourcePrefix, TextEncoderPrefix);
        var decoder = DecoderHyperParameters.Read(textSource);
        int layer = src.Get(Prefix + "text_encoder.layer", decoder.LogicalLayers - 2);
        var clip = ClipTextHyperParameters.Read(src, Prefix, "text_encoder_2");
        var graphs = new Dictionary<string, Graph.ComputeGraph>
        {
            [VideoDiffusionGraphs.TextEncode] = new DecoderGraphBuilder(textSource, new AdapterOptions { IncludeAuxiliaryGraphs = false }, decoder)
                .BuildLayerHiddenStates(VideoDiffusionGraphs.TextEncode, VideoDiffusionGraphs.Tokens, VideoDiffusionGraphs.Hidden, promptPrefix + promptLength, layer),
            [DiffusionGraphs.TextEncode2] = ClipTextGraphBuilder.Build(src.Weight, clip, SecondTextEncoderPrefix, DiffusionGraphs.TextEncode2, pooled: true, hidden: false),
            [VideoDiffusionGraphs.Denoise] = HunyuanVideoTransformerGraphBuilder.Build(src.Weight, transformer, latentFrames, latentHeight, latentWidth, promptLength),
            [VideoDiffusionGraphs.VaeDecode] = HunyuanVideoVaeGraphBuilder.BuildDecoder(src.Weight, latentChannels, src.Get(v + "scaling_factor", 0.476986f),
                src.Get(v + "groups", 32), temporal, spatial, latentFrames, latentHeight, latentWidth),
        };

        var attachments = new Dictionary<string, IWeightSource>(StringComparer.Ordinal);
        bool schedulerAttachment = FlowSchedulerComponent.AddAttachment(src, Prefix, attachments);
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = graphs,
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Attachments = attachments,
            Description = Describe(src, transformer, latentChannels, spatial, temporal, promptLength, promptPrefix, clip.ContextLength, schedulerAttachment),
        };
    }

    /// <summary>Fails with the list of missing parts when a transformer-only source comes without its companions.</summary>
    private static void RequireComponents(ModelSource src)
    {
        var missing = new List<string>();
        if (!src.Has(TextEncoderSourcePrefix + "architecture"))
        {
            missing.Add("text-encoder (the LLaVA-Llama-3 language model with its tokenizer)");
        }

        if (!src.Has(Prefix + "text_encoder_2.hidden_size"))
        {
            missing.Add("text-encoder-2 (CLIP-L with its tokenizer)");
        }

        if (!src.Has(Prefix + "vae.latent_channels"))
        {
            missing.Add("vae");
        }

        if (missing.Count > 0)
        {
            throw new NotSupportedException($"'{src.Path}' holds only the HunyuanVideo transformer; it needs its companions, given as auxiliary sources by role "
                + $"(a HunyuanVideo diffusers folder serves as all of them): {string.Join(", ", missing)}.");
        }
    }

    private static ModelDescription Describe(ModelSource src, HunyuanVideoHyperParameters transformer, int latentChannels, int spatial, int temporal, int promptLength,
        int promptPrefix, int clipLength, bool schedulerAttachment)
    {
        string v = Prefix + "vae.";
        var model = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.Width, src.Get(Prefix + "default_width", 1280))
            .Add(DiffusionConfigKeys.Height, src.Get(Prefix + "default_height", 720))
            .Add(VideoConfigKeys.Frames, src.Get(Prefix + "default_frames", 129))
            .Add(VideoConfigKeys.FramesPerSecond, src.Get(Prefix + "frames_per_second", 24.0))
            .Add(DiffusionConfigKeys.Steps, src.Get(Prefix + "default_steps", 50))
            .Add(DiffusionConfigKeys.GuidanceScale, src.Get(Prefix + "default_guidance", 6.0))
            .Add(VideoConfigKeys.EmbeddedGuidance, transformer.GuidanceEmbed);
        var textEncoder = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.ContextLength, promptLength)
            .Add(VideoConfigKeys.PromptPrefixTokens, promptPrefix);
        if (src.Get<string?>(Prefix + "text_encoder.prompt_template", null) is { } template)
        {
            textEncoder.Add(VideoConfigKeys.PromptTemplate, template);
        }

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
                    Config = textEncoder.Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, promptPrefix + promptLength).Build(),
                },
                new ModelComponent
                {
                    Role = DiffusionGraphs.SecondTextEncoderRole,
                    Graphs = [DiffusionGraphs.TextEncode2],
                    WeightPrefixes = [SecondTextEncoderPrefix],
                    Config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, clipLength).Build(),
                },
                new ModelComponent
                {
                    Role = DiffusionGraphs.SecondTokenizerRole,
                    MetadataPrefix = Prefix + "tokenizer_2.",
                    Config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, clipLength).Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Transformer,
                    Graphs = [VideoDiffusionGraphs.Denoise],
                    WeightPrefixes = [TransformerPrefix],
                    Config = new ComponentConfigBuilder()
                        .Add(DiffusionConfigKeys.LatentChannels, transformer.InChannels)
                        .Add(DiffusionConfigKeys.PatchSize, transformer.PatchSize)
                        .Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Vae,
                    Graphs = [VideoDiffusionGraphs.VaeDecode],
                    WeightPrefixes = [VaePrefix],
                    Config = new ComponentConfigBuilder()
                        .Add(DiffusionConfigKeys.LatentChannels, latentChannels)
                        .Add(DiffusionConfigKeys.Downscale, spatial)
                        .Add(VideoConfigKeys.TemporalDownscale, temporal)
                        .Add(VideoConfigKeys.DecodeTileFrames, src.Get(v + "tile_frames", 4))
                        .Add(VideoConfigKeys.DecodeTileStride, src.Get(v + "tile_stride", 3))
                        .Add(VideoConfigKeys.DecodeSpatialTile, src.Get(v + "spatial_tile", 32))
                        .Add(VideoConfigKeys.DecodeSpatialTileStride, src.Get(v + "spatial_tile_stride", 24))
                        .Build(),
                },
                FlowSchedulerComponent.Create(src, Prefix, schedulerAttachment, 7f),
            ],
            Config = model.Build(),
        };
    }
}
