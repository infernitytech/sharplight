using Sharplight.ImageGeneration;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Builds Z-Image generators (architecture <c>zimage</c>: Z-Image-Turbo and its fine-tunes) as an <see cref="ModelKinds.ImageGeneration"/>
/// model: a decoder language model (Qwen3-4B) whose second-to-last hidden states condition the image (the decoder graph of
/// <see cref="DecoderGraphBuilder"/> run over the chat-templated prompt), the single-stream DiT (<see cref="ZImageTransformerGraphBuilder"/>)
/// and the 16-channel FLUX autoencoder, plus the language model's tokenizer and a flow-matching scheduler component. The built-in
/// latent-diffusion pipeline runs it; the distilled Turbo model needs no guidance (8 steps).
/// </summary>
/// <remarks>
/// Canonical vocabulary: the transformer as documented on <see cref="ZImageTransformerGraphBuilder"/>; the text encoder's canonical decoder
/// weights under <c>text_encoder.</c>; the VAE as on <see cref="LatentDiffusionAdapter"/>. Metadata: <c>zimage.transformer.*</c>
/// (<see cref="ZImageHyperParameters"/>), <c>zimage.vae.*</c>, the text encoder's own canonical metadata under
/// <c>zimage.text_encoder_source.</c> (its architecture under <c>zimage.text_encoder_source.architecture</c>),
/// <c>zimage.text_encoder.prompt_length</c>, <c>zimage.text_encoder.layer</c> (the layer whose input is the conditioning, default the
/// last), <c>zimage.text_encoder.prompt_template</c>, the tokenizer under the canonical <c>tokenizer.*</c> keys, <c>zimage.scheduler.*</c>
/// (see <see cref="FlowSchedulerComponent"/>), <c>zimage.default_steps</c> and <c>zimage.default_guidance</c>.
/// A transformer-only source (as the GGUF files are distributed) gets the text encoder (any decoder source, such as a Qwen3-4B GGUF) and
/// the VAE as auxiliary sources (<see cref="ZImageCompanionAdapter"/>).
/// </remarks>
public sealed class ZImageAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "zimage";

    /// <summary>The metadata prefix of the text encoder's own canonical metadata.</summary>
    public const string TextEncoderSourcePrefix = Prefix + "text_encoder_source.";

    /// <summary>The canonical weight prefix of the text encoder.</summary>
    public const string TextEncoderPrefix = "text_encoder.";

    private const string Prefix = Architecture + ".";
    private const string TransformerPrefix = "transformer.";
    private const string VaePrefix = "vae.";

    /// <inheritdoc/>
    public string Name => "Z-Image (single-stream DiT + VAE + decoder language model)";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures { get; } = [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        ArgumentNullException.ThrowIfNull(sources);
        ArgumentNullException.ThrowIfNull(options);
        var src = sources.Primary;
        RequireComponents(src);
        var transformer = ZImageHyperParameters.Read(src, Prefix);
        var vae = AutoencoderHyperParameters.Read(src, Prefix);
        int promptLength = src.Get(Prefix + "text_encoder.prompt_length", 512);
        int downscale = vae.Downscale;
        long multiple = downscale * transformer.PatchSize;
        long height = options.Dimensions.GetValueOrDefault(DiffusionGraphs.Height, transformer.SampleSize);
        long width = options.Dimensions.GetValueOrDefault(DiffusionGraphs.Width, transformer.SampleSize);
        if (height <= 0 || width <= 0 || height % multiple != 0 || width % multiple != 0)
        {
            throw new ArgumentException($"Image size {width}x{height} must be a positive multiple of {multiple} pixels for {src.Get(CanonicalKeys.Name, src.Architecture)}.");
        }

        var textSource = TextEncoderSource(src);
        var decoder = DecoderHyperParameters.Read(textSource);
        int layer = src.Get(Prefix + "text_encoder.layer", decoder.LogicalLayers - 1);
        var textGraph = new DecoderGraphBuilder(textSource, new AdapterOptions { IncludeAuxiliaryGraphs = false }, decoder)
            .BuildLayerHiddenStates(DiffusionGraphs.TextEncode, DiffusionGraphs.Tokens, DiffusionGraphs.Hidden, promptLength, layer);
        long latentHeight = height / downscale;
        long latentWidth = width / downscale;
        var graphs = new Dictionary<string, Graph.ComputeGraph>
        {
            [DiffusionGraphs.TextEncode] = textGraph,
            [DiffusionGraphs.Denoise] = ZImageTransformerGraphBuilder.Build(src.Weight, transformer, latentHeight, latentWidth, promptLength),
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
            Description = Describe(src, transformer, vae, promptLength, schedulerAttachment),
        };
    }

    /// <summary>The text encoder as a decoder source of its own: its canonical metadata and its weights without the prefix.</summary>
    private static ModelSource TextEncoderSource(ModelSource src)
    {
        var metadata = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var (key, value) in src.Metadata)
        {
            if (key.StartsWith(TextEncoderSourcePrefix, StringComparison.Ordinal))
            {
                metadata[key[TextEncoderSourcePrefix.Length..]] = value;
            }
        }

        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var (name, weight) in src.Weights)
        {
            if (name.StartsWith(TextEncoderPrefix, StringComparison.Ordinal))
            {
                weights[name[TextEncoderPrefix.Length..]] = weight;
            }
        }

        string architecture = metadata.TryGetValue("architecture", out var stored) ? stored.ToString()! : "decoder";
        return new ModelSource
        {
            Format = src.Format,
            Architecture = architecture,
            Metadata = metadata,
            Weights = weights,
            Path = src.Path,
        };
    }

    /// <summary>Fails with the list of missing parts when a transformer-only source comes without its companions.</summary>
    private static void RequireComponents(ModelSource src)
    {
        var missing = new List<string>();
        if (!src.Has(TextEncoderSourcePrefix + "architecture"))
        {
            missing.Add("text-encoder (Qwen3-4B)");
        }

        if (!src.Has(Prefix + "vae.block_channels"))
        {
            missing.Add("vae");
        }

        if (missing.Count > 0)
        {
            throw new NotSupportedException($"'{src.Path}' holds only the Z-Image transformer; it needs its companion files, given as auxiliary sources by role "
                + $"(or placed next to it): {string.Join(", ", missing)}.");
        }
    }

    private static ModelDescription Describe(ModelSource src, ZImageHyperParameters transformer, AutoencoderHyperParameters vae, int promptLength,
        bool hasSchedulerAttachment)
    {
        var vaeConfig = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.LatentChannels, vae.LatentChannels)
            .Add(DiffusionConfigKeys.Downscale, vae.Downscale);
        if (vae.LatentRgbFactors.Length > 0)
        {
            vaeConfig.Add(DiffusionConfigKeys.LatentRgbFactors, vae.LatentRgbFactors);
        }

        var tokenizer = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, promptLength);
        if (src.Get<string?>(Prefix + "text_encoder.prompt_template", null) is { } template)
        {
            tokenizer.Add(DiffusionConfigKeys.PromptTemplate, template);
        }

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
                    Config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, promptLength).Build(),
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
                        .Build(),
                    SymbolBounds = new Dictionary<string, long> { [DiffusionGraphs.BatchSymbol] = batch1 },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Vae,
                    Graphs = [DiffusionGraphs.VaeDecode, DiffusionGraphs.VaeEncode],
                    WeightPrefixes = [VaePrefix],
                    Config = vaeConfig.Build(),
                    SymbolBounds = new Dictionary<string, long> { [DiffusionGraphs.BatchSymbol] = batch1 },
                },
                FlowSchedulerComponent.Create(src, Prefix, hasSchedulerAttachment, defaultShift: 3f),
            ],
            Config = new ComponentConfigBuilder()
                .Add(DiffusionConfigKeys.Width, transformer.SampleSize)
                .Add(DiffusionConfigKeys.Height, transformer.SampleSize)
                .Add(DiffusionConfigKeys.Steps, src.Get(Prefix + "default_steps", 8))
                .Add(DiffusionConfigKeys.GuidanceScale, src.Get(Prefix + "default_guidance", 0.0))
                .Add(DiffusionConfigKeys.ClassifierFreeGuidance, src.Get(Prefix + "classifier_free_guidance", false))
                .Build(),
        };
    }
}
