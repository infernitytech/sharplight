using Sharplight.ImageGeneration;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Builds Qwen-Image generators (architecture <c>qwen_image</c>: Qwen-Image and its fine-tunes) as an <see cref="ModelKinds.ImageGeneration"/>
/// model: a decoder language model (Qwen2.5-VL-7B's text model) whose final-normed last hidden states over the system-templated prompt
/// condition the image (the decoder graph of <see cref="DecoderGraphBuilder"/>; the template's leading tokens are dropped by the
/// pipeline, <see cref="DiffusionConfigKeys.PromptSkip"/>), the dual-stream MMDiT (<see cref="FluxTransformerGraphBuilder"/> without
/// single-stream blocks: RMS-normed text input, separate q/k/v, centred RoPE, masked text padding) and the causal 3-D VAE of Wan 2.1 for
/// one frame (<see cref="WanVaeGraphBuilder"/>), plus the language model's tokenizer and a flow-matching scheduler component.
/// Classifier-free guidance rescales the guided prediction to the conditional one's norm per token
/// (<see cref="DiffusionConfigKeys.GuidanceRenormalization"/>).
/// </summary>
/// <remarks>
/// Canonical vocabulary: the transformer as documented on <see cref="FluxTransformerGraphBuilder"/>; the text encoder's canonical decoder
/// weights under <c>text_encoder.</c>; the VAE as on <see cref="WanVaeGraphBuilder"/>. Metadata: <c>qwen_image.transformer.*</c>
/// (<see cref="FluxHyperParameters"/>), <c>qwen_image.vae.*</c> (<see cref="WanVaeHyperParameters"/>), the text encoder's own canonical
/// metadata under <c>qwen_image.text_encoder_source.</c> (its architecture under <c>qwen_image.text_encoder_source.architecture</c>),
/// <c>qwen_image.text_encoder.prompt_length</c>, <c>.prompt_skip</c>, <c>.prompt_template</c>, the tokenizer under the canonical
/// <c>tokenizer.*</c> keys, <c>qwen_image.scheduler.*</c> (see <see cref="FlowSchedulerComponent"/>), <c>qwen_image.default_steps</c> and
/// <c>qwen_image.default_guidance</c>. A transformer-only source (as the GGUF files are distributed) gets the text encoder (any decoder
/// source, such as a Qwen2.5-VL-7B GGUF) and the VAE as auxiliary sources (<see cref="QwenImageCompanionAdapter"/>).
/// </remarks>
public sealed class QwenImageAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "qwen_image";

    /// <summary>The metadata prefix of the text encoder's own canonical metadata.</summary>
    public const string TextEncoderSourcePrefix = Prefix + "text_encoder_source.";

    /// <summary>The canonical weight prefix of the text encoder.</summary>
    public const string TextEncoderPrefix = "text_encoder.";

    private const string Prefix = Architecture + ".";
    private const string TransformerPrefix = "transformer.";
    private const string VaePrefix = "vae.";

    /// <inheritdoc/>
    public string Name => "Qwen-Image (dual-stream MMDiT + causal VAE + decoder language model)";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures { get; } = [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        ArgumentNullException.ThrowIfNull(sources);
        ArgumentNullException.ThrowIfNull(options);
        var src = sources.Primary;
        RequireComponents(src);
        var transformer = FluxHyperParameters.Read(src, Prefix);
        var vae = WanVaeHyperParameters.Read(src, Prefix);
        int promptLength = src.Get(Prefix + "text_encoder.prompt_length", 512);
        int promptSkip = src.Get(Prefix + "text_encoder.prompt_skip", 0);
        int downscale = vae.SpatialDownscale;
        long multiple = downscale * transformer.PatchSize;
        long height = options.Dimensions.GetValueOrDefault(DiffusionGraphs.Height, transformer.SampleSize);
        long width = options.Dimensions.GetValueOrDefault(DiffusionGraphs.Width, transformer.SampleSize);
        if (height <= 0 || width <= 0 || height % multiple != 0 || width % multiple != 0)
        {
            throw new ArgumentException($"Image size {width}x{height} must be a positive multiple of {multiple} pixels for {src.Get(CanonicalKeys.Name, src.Architecture)}.");
        }

        var textSource = DecoderTextEncoder.Source(src, TextEncoderSourcePrefix, TextEncoderPrefix);
        var decoder = DecoderHyperParameters.Read(textSource);
        var textGraph = new DecoderGraphBuilder(textSource, new AdapterOptions { IncludeAuxiliaryGraphs = false }, decoder)
            .BuildLayerHiddenStates(DiffusionGraphs.TextEncode, DiffusionGraphs.Tokens, DiffusionGraphs.Hidden, promptSkip + promptLength, decoder.LogicalLayers,
                finalNorm: true);
        long latentHeight = height / downscale;
        long latentWidth = width / downscale;
        var graphs = new Dictionary<string, Graph.ComputeGraph>
        {
            [DiffusionGraphs.TextEncode] = textGraph,
            [DiffusionGraphs.Denoise] = FluxTransformerGraphBuilder.Build(src.Weight, transformer, latentHeight, latentWidth, promptLength),
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
            missing.Add("text-encoder (Qwen2.5-VL-7B)");
        }

        if (!src.Has(Prefix + "vae.latents_mean"))
        {
            missing.Add("vae");
        }

        if (missing.Count > 0)
        {
            throw new NotSupportedException($"'{src.Path}' holds only the Qwen-Image transformer; it needs its companion files, given as auxiliary sources by role "
                + $"(or placed next to it): {string.Join(", ", missing)}.");
        }
    }

    private static ModelDescription Describe(ModelSource src, FluxHyperParameters transformer, WanVaeHyperParameters vae, int promptLength, int promptSkip,
        bool hasSchedulerAttachment)
    {
        var tokenizer = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, promptSkip + promptLength);
        if (src.Get<string?>(Prefix + "text_encoder.prompt_template", null) is { } template)
        {
            tokenizer.Add(DiffusionConfigKeys.PromptTemplate, template);
        }

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
                        .Add(DiffusionConfigKeys.PatchSize, transformer.PatchSize)
                        .Add(DiffusionConfigKeys.ContextMaskPadding, 0)
                        .Add(DiffusionConfigKeys.GuidanceRenormalization, true)
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
                .Add(DiffusionConfigKeys.Steps, src.Get(Prefix + "default_steps", 50))
                .Add(DiffusionConfigKeys.GuidanceScale, src.Get(Prefix + "default_guidance", 4.0))
                .Add(DiffusionConfigKeys.ClassifierFreeGuidance, true)
                .Build(),
        };
    }
}
