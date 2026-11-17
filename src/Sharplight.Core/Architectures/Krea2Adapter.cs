using Sharplight.ImageGeneration;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Builds Krea 2 generators (architecture <c>krea2</c>: Krea 2 Turbo and Raw) as an <see cref="ModelKinds.ImageGeneration"/> model: a decoder
/// language model (Qwen3-VL-4B's text model) whose hidden states entering twelve layers (2, 5, …, 35), stacked per token, condition the image
/// (the system-templated prompt, the template's system turn dropped, <see cref="DiffusionConfigKeys.PromptSkip"/>), the single-stream DiT
/// with its text fusion (<see cref="Krea2TransformerGraphBuilder"/>) on 2×2 patches of 16-channel latents, and the Qwen-Image VAE (the Wan 2.1
/// causal VAE for one frame, <see cref="WanVaeGraphBuilder"/>), plus the language model's tokenizer and a flow-matching scheduler component.
/// Turbo runs 8 steps without guidance; Raw 28 steps with classifier-free guidance.
/// </summary>
/// <remarks>
/// Canonical vocabulary: the transformer as documented on <see cref="Krea2TransformerGraphBuilder"/>; the text encoder's canonical decoder
/// weights under <c>text_encoder.</c>; the VAE as on <see cref="WanVaeGraphBuilder"/>. Metadata: <c>krea2.transformer.*</c>
/// (<see cref="Krea2HyperParameters"/>), <c>krea2.vae.*</c> (<see cref="WanVaeHyperParameters"/>), the text encoder's own canonical metadata
/// under <c>krea2.text_encoder_source.</c>, <c>krea2.text_encoder.prompt_length</c>, <c>.prompt_skip</c>, <c>.prompt_template</c>,
/// <c>.layers</c>, the tokenizer under the canonical <c>tokenizer.*</c> keys, <c>krea2.scheduler.*</c>, <c>krea2.default_steps</c>,
/// <c>krea2.default_guidance</c> (the usual scale: diffusers' Krea convention cond + g · (cond − uncond) is 1 + g) and
/// <c>krea2.classifier_free_guidance</c>. Transformer-only files get the text encoder and the VAE as auxiliary sources
/// (<see cref="Krea2CompanionAdapter"/>).
/// </remarks>
public sealed class Krea2Adapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "krea2";

    /// <summary>The metadata prefix of the text encoder's own canonical metadata.</summary>
    public const string TextEncoderSourcePrefix = Prefix + "text_encoder_source.";

    /// <summary>The canonical weight prefix of the text encoder.</summary>
    public const string TextEncoderPrefix = "text_encoder.";

    private const string Prefix = Architecture + ".";
    private const string TransformerPrefix = "transformer.";
    private const string VaePrefix = "vae.";

    /// <inheritdoc/>
    public string Name => "Krea 2 (single-stream DiT with text fusion + Qwen-Image VAE + decoder language model)";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures { get; } = [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        ArgumentNullException.ThrowIfNull(sources);
        ArgumentNullException.ThrowIfNull(options);
        var src = sources.Primary;
        RequireComponents(src);
        var transformer = Krea2HyperParameters.Read(src, Prefix);
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
        // Transformer-only files do not say which layers the pipeline taps: every (layers / tapped)-th, ending one before the last, as the
        // released pipelines (2, 5, …, 35 of Qwen3-VL-4B's 36).
        int stride = Math.Max(1, decoder.LogicalLayers / transformer.TextLayers);
        int[] layers = src.Get(Prefix + "text_encoder.layers", Enumerable.Range(0, transformer.TextLayers).Select(i => stride - 1 + i * stride).ToArray());
        if (layers.Length != transformer.TextLayers)
        {
            throw new InvalidDataException($"The Krea 2 transformer takes {transformer.TextLayers} text-encoder layers, the pipeline taps {layers.Length}.");
        }

        var textGraph = new DecoderGraphBuilder(textSource, new AdapterOptions { IncludeAuxiliaryGraphs = false }, decoder)
            .BuildLayerHiddenStates(DiffusionGraphs.TextEncode, DiffusionGraphs.Tokens, DiffusionGraphs.Hidden, promptSkip + promptLength, layers);
        long latentHeight = height / downscale;
        long latentWidth = width / downscale;
        var graphs = new Dictionary<string, Graph.ComputeGraph>
        {
            [DiffusionGraphs.TextEncode] = textGraph,
            [DiffusionGraphs.Denoise] = Krea2TransformerGraphBuilder.Build(src.Weight, transformer, latentHeight, latentWidth, promptLength),
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
            missing.Add("text-encoder (Qwen3-VL-4B)");
        }

        if (!src.Has(Prefix + "vae.latents_mean"))
        {
            missing.Add("vae");
        }

        if (missing.Count > 0)
        {
            throw new NotSupportedException($"'{src.Path}' holds only the Krea 2 transformer; it needs its companion files, given as auxiliary sources by role "
                + $"(or placed next to it): {string.Join(", ", missing)}.");
        }
    }

    private static ModelDescription Describe(ModelSource src, Krea2HyperParameters transformer, WanVaeHyperParameters vae, int promptLength, int promptSkip,
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
                .Add(DiffusionConfigKeys.Steps, src.Get(Prefix + "default_steps", 8))
                .Add(DiffusionConfigKeys.GuidanceScale, src.Get(Prefix + "default_guidance", 1.0))
                .Add(DiffusionConfigKeys.ClassifierFreeGuidance, src.Get(Prefix + "classifier_free_guidance", false))
                .Build(),
        };
    }
}
