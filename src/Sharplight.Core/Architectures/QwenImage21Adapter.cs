using Sharplight.ImageGeneration;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Builds Qwen-Image 2.1 generators (architecture <c>qwen_image21</c>: Qwen-Image 2.1 and its Turbo) as an
/// <see cref="ModelKinds.ImageGeneration"/> model: a decoder language model (Qwen3-VL-8B's text model) whose last hidden states before the
/// final norm over the system-templated prompt condition the image (the template's system turn dropped,
/// <see cref="DiffusionConfigKeys.PromptSkip"/>), the single-stream block-causal DiT (<see cref="QwenImage21TransformerGraphBuilder"/>) and the
/// 16× residual causal VAE with RGBA input / output for one frame (<see cref="WanVaeGraphBuilder"/>), plus the language model's tokenizer
/// and a flow-matching scheduler component. Text-to-image only (the image-conditioned editing path feeds Qwen3-VL image tokens).
/// </summary>
/// <remarks>
/// Canonical vocabulary: the transformer as documented on <see cref="QwenImage21TransformerGraphBuilder"/>; the text encoder's canonical
/// decoder weights under <c>text_encoder.</c>; the VAE as on <see cref="WanVaeGraphBuilder"/>. Metadata: <c>qwen_image21.transformer.*</c>
/// (<see cref="QwenImage21HyperParameters"/>), <c>qwen_image21.vae.*</c> (<see cref="WanVaeHyperParameters"/>), the text encoder's own
/// canonical metadata under <c>qwen_image21.text_encoder_source.</c>, <c>qwen_image21.text_encoder.prompt_length</c>, <c>.prompt_skip</c>,
/// <c>.prompt_template</c>, the tokenizer under the canonical <c>tokenizer.*</c> keys, <c>qwen_image21.scheduler.*</c>,
/// <c>qwen_image21.default_steps</c> and <c>qwen_image21.default_guidance</c>. Transformer-only files get the text encoder and the VAE as
/// auxiliary sources (<see cref="QwenImage21CompanionAdapter"/>).
/// </remarks>
public sealed class QwenImage21Adapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "qwen_image21";

    /// <summary>The metadata prefix of the text encoder's own canonical metadata.</summary>
    public const string TextEncoderSourcePrefix = Prefix + "text_encoder_source.";

    /// <summary>The canonical weight prefix of the text encoder.</summary>
    public const string TextEncoderPrefix = "text_encoder.";

    private const string Prefix = Architecture + ".";
    private const string TransformerPrefix = "transformer.";
    private const string VaePrefix = "vae.";

    /// <inheritdoc/>
    public string Name => "Qwen-Image 2.1 (single-stream block-causal DiT + residual VAE + decoder language model)";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures { get; } = [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        ArgumentNullException.ThrowIfNull(sources);
        ArgumentNullException.ThrowIfNull(options);
        var src = sources.Primary;
        RequireComponents(src);
        var transformer = QwenImage21HyperParameters.Read(src, Prefix);
        var vae = WanVaeHyperParameters.Read(src, Prefix);
        int promptLength = src.Get(Prefix + "text_encoder.prompt_length", 512);
        int promptSkip = src.Get(Prefix + "text_encoder.prompt_skip", 0);
        int downscale = vae.SpatialDownscale;
        // The pipeline lays the image tokens out in 2×2 groups of the language model's image slots: sizes are multiples of two tokens.
        long multiple = downscale * 2;
        long height = options.Dimensions.GetValueOrDefault(DiffusionGraphs.Height, transformer.SampleSize);
        long width = options.Dimensions.GetValueOrDefault(DiffusionGraphs.Width, transformer.SampleSize);
        if (height <= 0 || width <= 0 || height % multiple != 0 || width % multiple != 0)
        {
            throw new ArgumentException($"Image size {width}x{height} must be a positive multiple of {multiple} pixels for {src.Get(CanonicalKeys.Name, src.Architecture)}.");
        }

        var textSource = DecoderTextEncoder.Source(src, TextEncoderSourcePrefix, TextEncoderPrefix);
        var decoder = DecoderHyperParameters.Read(textSource);
        var textGraph = new DecoderGraphBuilder(textSource, new AdapterOptions { IncludeAuxiliaryGraphs = false }, decoder)
            .BuildLayerHiddenStates(DiffusionGraphs.TextEncode, DiffusionGraphs.Tokens, DiffusionGraphs.Hidden, promptSkip + promptLength, decoder.LogicalLayers);
        long latentHeight = height / downscale;
        long latentWidth = width / downscale;
        var graphs = new Dictionary<string, Graph.ComputeGraph>
        {
            [DiffusionGraphs.TextEncode] = textGraph,
            [DiffusionGraphs.Denoise] = QwenImage21TransformerGraphBuilder.Build(src.Weight, transformer, latentHeight, latentWidth, promptLength),
            [DiffusionGraphs.VaeDecode] = WanVaeGraphBuilder.BuildImageDecoder(src.Weight, vae, latentHeight, latentWidth),
            [DiffusionGraphs.VaeEncode] = WanVaeGraphBuilder.BuildImageEncoder(src.Weight, vae, height, width),
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
            Description = Describe(src, transformer, vae, promptLength, promptSkip, schedulerAttachment),
        };
    }

    /// <summary>Fails with the list of missing parts when a transformer-only source comes without its companions.</summary>
    private static void RequireComponents(ModelSource src)
    {
        var missing = new List<string>();
        if (!src.Has(TextEncoderSourcePrefix + "architecture"))
        {
            missing.Add("text-encoder (Qwen3-VL-8B)");
        }

        if (!src.Has(Prefix + "vae.latents_mean"))
        {
            missing.Add("vae");
        }

        if (missing.Count > 0)
        {
            throw new NotSupportedException($"'{src.Path}' holds only the Qwen-Image 2.1 transformer; it needs its companion files, given as auxiliary sources by role "
                + $"(or placed next to it): {string.Join(", ", missing)}.");
        }
    }

    private static ModelDescription Describe(ModelSource src, QwenImage21HyperParameters transformer, WanVaeHyperParameters vae, int promptLength, int promptSkip,
        bool hasSchedulerAttachment)
    {
        var tokenizer = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, promptSkip + promptLength);
        if (src.Get<string?>(Prefix + "text_encoder.prompt_template", null) is { } template)
        {
            tokenizer.Add(DiffusionConfigKeys.PromptTemplate, template);
        }

        tokenizer.Add(DiffusionConfigKeys.EmptyPrompt, " ");

        var scheduler = FlowSchedulerComponent.Create(src, Prefix, hasSchedulerAttachment);
        long batch1 = 1;
        return new ModelDescription
        {
            Kind = ModelKinds.ImageGeneration,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.TextEncoder,
                    Graphs = [DiffusionGraphs.TextEncode],
                    WeightPrefixes = [TextEncoderPrefix],
                    Config = new ComponentConfigBuilder()
                        .Add(DiffusionConfigKeys.ContextLength, promptSkip + promptLength)
                        .Add(DiffusionConfigKeys.PromptSkip, promptSkip)
                        .Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Config = tokenizer.Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Denoiser,
                    Graphs = [DiffusionGraphs.Denoise],
                    WeightPrefixes = [TransformerPrefix],
                    Config = new ComponentConfigBuilder()
                        .Add(DiffusionConfigKeys.LatentChannels, transformer.LatentChannels)
                        .Add(DiffusionConfigKeys.PatchSize, 1)
                        .Add(DiffusionConfigKeys.ContextMaskPadding, 0)
                        .Build(),
                    SymbolBounds = new Dictionary<string, long> { [DiffusionGraphs.BatchSymbol] = batch1 },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Vae,
                    Graphs = [DiffusionGraphs.VaeDecode, DiffusionGraphs.VaeEncode],
                    WeightPrefixes = [VaePrefix],
                    Config = new ComponentConfigBuilder()
                        .Add(DiffusionConfigKeys.LatentChannels, vae.LatentChannels)
                        .Add(DiffusionConfigKeys.Downscale, vae.SpatialDownscale)
                        .Build(),
                    SymbolBounds = new Dictionary<string, long> { [DiffusionGraphs.BatchSymbol] = batch1 },
                },
                scheduler,
            ],
            Config = new ComponentConfigBuilder()
                .Add(DiffusionConfigKeys.Width, transformer.SampleSize)
                .Add(DiffusionConfigKeys.Height, transformer.SampleSize)
                .Add(DiffusionConfigKeys.Steps, src.Get(Prefix + "default_steps", 40))
                .Add(DiffusionConfigKeys.GuidanceScale, src.Get(Prefix + "default_guidance", 1.0))
                .Add(DiffusionConfigKeys.ClassifierFreeGuidance, true)
                .Build(),
        };
    }
}
