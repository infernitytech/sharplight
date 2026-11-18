using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Vision;

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
        int maxReferences = src.Get(Prefix + "edit.max_references", 0);
        var vision = maxReferences > 0 ? VisionParameters.FromMetadata(textSource.Metadata, textSource.Architecture + ".vision.") : null;
        if (maxReferences > 0 && vision is null)
        {
            throw new NotSupportedException($"'{src.Path}' is a Qwen-Image editing model, whose text encoder reads the reference images: its Qwen2.5-VL text encoder needs the "
                + "vision tower (the pipeline's text_encoder folder holds it; a text-only language model file does not).");
        }

        var textBuilder = new DecoderGraphBuilder(textSource, new AdapterOptions { IncludeAuxiliaryGraphs = false }, decoder);
        var textGraph = textBuilder.BuildLayerHiddenStates(DiffusionGraphs.TextEncode, DiffusionGraphs.Tokens, DiffusionGraphs.Hidden, promptSkip + promptLength,
            [decoder.LogicalLayers], finalNorm: true, imageRows: vision is not null);
        long latentHeight = height / downscale;
        long latentWidth = width / downscale;
        var graphs = new Dictionary<string, Graph.ComputeGraph>
        {
            [DiffusionGraphs.TextEncode] = textGraph,
            [DiffusionGraphs.Denoise] = FluxTransformerGraphBuilder.Build(src.Weight, transformer, latentHeight, latentWidth, promptLength,
                options.Dimensions.GetValueOrDefault(DiffusionGraphs.References, 0) is var references and > 0 ? (int)references : 0),
            [DiffusionGraphs.VaeDecode] = WanVaeGraphBuilder.BuildImageDecoder(src.Weight, vae, latentHeight, latentWidth),
            [DiffusionGraphs.VaeEncode] = WanVaeGraphBuilder.BuildImageEncoder(src.Weight, vae, height, width),
        };

        var properties = new Dictionary<string, object>(StringComparer.Ordinal);
        if (vision is not null)
        {
            // The vision tower of the text encoder and what the image-prompt assembly (Qwen35Multimodal) reads.
            graphs[WellKnown.VisionEncodeGraph] = VisionTowerGraph.Build(vision, textSource.Weight);
            vision.WriteProperties(properties);
            properties[WellKnown.PositionAxes] = decoder.PositionAxes;
        }

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
            Properties = properties,
            Description = Describe(src, transformer, vae, promptLength, promptSkip, schedulerAttachment, vision is not null),
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
        bool hasSchedulerAttachment, bool edit)
    {
        var tokenizer = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, promptSkip + promptLength);
        var textEncoder = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.ContextLength, promptSkip + promptLength)
            .Add(DiffusionConfigKeys.PromptSkip, promptSkip);
        var denoiserConfig = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.LatentChannels, transformer.LatentChannels)
            .Add(DiffusionConfigKeys.PatchSize, transformer.PatchSize)
            .Add(DiffusionConfigKeys.ContextMaskPadding, 0)
            .Add(DiffusionConfigKeys.GuidanceRenormalization, true);
        var modelConfig = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.Width, transformer.SampleSize)
            .Add(DiffusionConfigKeys.Height, transformer.SampleSize)
            .Add(DiffusionConfigKeys.Steps, src.Get(Prefix + "default_steps", 50))
            .Add(DiffusionConfigKeys.GuidanceScale, src.Get(Prefix + "default_guidance", 4.0))
            .Add(DiffusionConfigKeys.ClassifierFreeGuidance, true);
        if (edit)
        {
            tokenizer.Add(DiffusionConfigKeys.ImagePrompt, src.Get(Prefix + "edit.image_prompt", ""));
            textEncoder.Add(DiffusionConfigKeys.ConditionPixels, src.Get(Prefix + "edit.condition_pixels", 384 * 384));
            denoiserConfig.Add(DiffusionConfigKeys.ReferenceImages, transformer.MaxReferences);
            modelConfig.Add(DiffusionConfigKeys.ReferenceArea, src.Get(Prefix + "edit.reference_area", 1024 * 1024));
        }

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
                    Graphs = edit ? [DiffusionGraphs.TextEncode, WellKnown.VisionEncodeGraph] : [DiffusionGraphs.TextEncode],
                    WeightPrefixes = [TextEncoderPrefix],
                    Config = textEncoder.Build(),
                    SymbolBounds = edit
                        ? new Dictionary<string, long>
                        {
                            [VisionParameters.PatchSymbol] = 4L * promptLength,
                            [VisionParameters.MergedSymbol] = promptLength,
                        }
                        : new Dictionary<string, long>(),
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
                    Config = denoiserConfig.Build(),
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
            Config = modelConfig.Build(),
        };
    }
}
