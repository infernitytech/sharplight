using Sharplight.ImageGeneration;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Builds FLUX.2 generators (architecture <c>flux2</c>: FLUX.2 Klein 4B / 9B and their base models) as an
/// <see cref="ModelKinds.ImageGeneration"/> model: a decoder language model (Qwen3) whose hidden states entering three layers (9, 18 and 27),
/// concatenated per token, condition the image (<see cref="DecoderGraphBuilder"/> over the chat-templated prompt of 512 tokens), the MMDiT
/// (<see cref="FluxTransformerGraphBuilder"/> with shared modulations, SwiGLU feed-forwards and 4-axis RoPE) on 128-channel latents, and the
/// 32-channel KL autoencoder whose latents are patchified 2×2 and normalized by its batch-norm statistics
/// (<see cref="AutoencoderGraphBuilder"/>), plus the language model's tokenizer and a flow-matching scheduler component with FLUX.2's
/// empirical shift. Distilled Klein models run 4 steps without guidance.
/// </summary>
/// <remarks>
/// Canonical vocabulary: the transformer as documented on <see cref="FluxTransformerGraphBuilder"/> (plus <c>transformer.mod_img</c>,
/// <c>mod_txt</c>, <c>mod_single</c>); the text encoder's canonical decoder weights under <c>text_encoder.</c>; the VAE as on
/// <see cref="AutoencoderGraphBuilder"/> with <c>vae.bn.running_mean|running_var</c>. Metadata: <c>flux2.transformer.*</c>
/// (<see cref="FluxHyperParameters"/>), <c>flux2.vae.*</c> (<see cref="AutoencoderHyperParameters"/>, <c>latent_patch</c> 2), the text
/// encoder's own canonical metadata under <c>flux2.text_encoder_source.</c>, <c>flux2.text_encoder.prompt_length</c>, <c>.layers</c>,
/// <c>.prompt_template</c>, <c>flux2.scheduler.*</c>, <c>flux2.default_steps</c>, <c>flux2.default_guidance</c> and
/// <c>flux2.classifier_free_guidance</c>. Transformer-only files get the text encoder and the VAE as auxiliary sources
/// (<see cref="Flux2CompanionAdapter"/>).
/// </remarks>
public sealed class Flux2Adapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "flux2";

    /// <summary>The metadata prefix of the text encoder's own canonical metadata.</summary>
    public const string TextEncoderSourcePrefix = Prefix + "text_encoder_source.";

    /// <summary>The canonical weight prefix of the text encoder.</summary>
    public const string TextEncoderPrefix = "text_encoder.";

    private const string Prefix = Architecture + ".";
    private const string TransformerPrefix = "transformer.";
    private const string VaePrefix = "vae.";

    /// <inheritdoc/>
    public string Name => "FLUX.2 (MMDiT + patchified KL VAE + decoder language model)";

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
        var vae = AutoencoderHyperParameters.Read(src, Prefix);
        int promptLength = src.Get(Prefix + "text_encoder.prompt_length", 512);
        int promptSkip = 0;
        int downscale = vae.ModelDownscale;
        long multiple = downscale * transformer.PatchSize;
        long height = options.Dimensions.GetValueOrDefault(DiffusionGraphs.Height, transformer.SampleSize);
        long width = options.Dimensions.GetValueOrDefault(DiffusionGraphs.Width, transformer.SampleSize);
        if (height <= 0 || width <= 0 || height % multiple != 0 || width % multiple != 0)
        {
            throw new ArgumentException($"Image size {width}x{height} must be a positive multiple of {multiple} pixels for {src.Get(CanonicalKeys.Name, src.Architecture)}.");
        }

        var textSource = DecoderTextEncoder.Source(src, TextEncoderSourcePrefix, TextEncoderPrefix);
        var decoder = DecoderHyperParameters.Read(textSource);
        int[] layers = src.Get<int[]>(Prefix + "text_encoder.layers", [9, 18, 27]);
        var textGraph = new DecoderGraphBuilder(textSource, new AdapterOptions { IncludeAuxiliaryGraphs = false }, decoder)
            .BuildLayerHiddenStates(DiffusionGraphs.TextEncode, DiffusionGraphs.Tokens, DiffusionGraphs.Hidden, promptLength, layers, maskInput: DiffusionGraphs.TokenMask,
                f32Activations: src.Get(Prefix + "text_encoder.f32_activations", false));
        long latentHeight = height / downscale;
        long latentWidth = width / downscale;
        var graphs = new Dictionary<string, Graph.ComputeGraph>
        {
            [DiffusionGraphs.TextEncode] = textGraph,
            [DiffusionGraphs.Denoise] = FluxTransformerGraphBuilder.Build(src.Weight, transformer, latentHeight, latentWidth, promptLength,
                (int)options.Dimensions.GetValueOrDefault(DiffusionGraphs.References, 0)),
            [DiffusionGraphs.VaeDecode] = AutoencoderGraphBuilder.BuildDecoder(src.Weight, vae, latentHeight, latentWidth),
            [DiffusionGraphs.VaeEncode] = AutoencoderGraphBuilder.BuildEncoder(src.Weight, vae, height, width),
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
            missing.Add("text-encoder (Qwen3, or Mistral Small 3.2 24B for FLUX.2 Dev)");
        }

        if (!src.Has(Prefix + "vae.block_channels"))
        {
            missing.Add("vae");
        }

        if (missing.Count > 0)
        {
            throw new NotSupportedException($"'{src.Path}' holds only the FLUX.2 transformer; it needs its companion files, given as auxiliary sources by role "
                + $"(or placed next to it): {string.Join(", ", missing)}.");
        }
    }

    private static ModelDescription Describe(ModelSource src, FluxHyperParameters transformer, AutoencoderHyperParameters vae, int promptLength, int promptSkip,
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
                        .Add(DiffusionConfigKeys.ContextLength, promptLength)
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
                        .Add(DiffusionConfigKeys.ReferenceImages, transformer.MaxReferences)
                        .Build(),
                    SymbolBounds = new Dictionary<string, long> { [DiffusionGraphs.BatchSymbol] = batch1 },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Vae,
                    Graphs = [DiffusionGraphs.VaeDecode, DiffusionGraphs.VaeEncode],
                    WeightPrefixes = [VaePrefix],
                    Config = new ComponentConfigBuilder()
                        .Add(DiffusionConfigKeys.LatentChannels, vae.ModelLatentChannels)
                        .Add(DiffusionConfigKeys.Downscale, vae.ModelDownscale)
                        .Build(),
                    SymbolBounds = new Dictionary<string, long> { [DiffusionGraphs.BatchSymbol] = batch1 },
                },
                scheduler,
            ],
            Config = new ComponentConfigBuilder()
                .Add(DiffusionConfigKeys.Width, transformer.SampleSize)
                .Add(DiffusionConfigKeys.Height, transformer.SampleSize)
                .Add(DiffusionConfigKeys.Steps, src.Get(Prefix + "default_steps", 4))
                .Add(DiffusionConfigKeys.GuidanceScale, src.Get(Prefix + "default_guidance", 1.0))
                .Add(DiffusionConfigKeys.ClassifierFreeGuidance, src.Get(Prefix + "classifier_free_guidance", false))
                .Build(),
        };
    }
}
