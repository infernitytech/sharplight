using Sharplight.ImageGeneration;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Builds Chroma image generators (architecture <c>chroma</c>: Chroma1-HD / Base / Flash and their fine-tunes, FLUX.1-schnell
/// derivatives) as an <see cref="ModelKinds.ImageGeneration"/> model: a T5 encoder run with the prompt's attention mask (no CLIP), the
/// FLUX MMDiT (<see cref="FluxTransformerGraphBuilder"/>) whose modulation comes from the "distilled guidance" approximator and whose
/// attention masks the text padding except its first token, and the 16-channel FLUX autoencoder, plus the T5 tokenizer and a
/// flow-matching scheduler component. The built-in latent-diffusion pipeline runs it with classifier-free guidance (one denoiser row
/// per run).
/// </summary>
/// <remarks>
/// Canonical vocabulary: the transformer as documented on <see cref="FluxTransformerGraphBuilder"/> without the modulation projections
/// and embedders, plus <c>transformer.approximator.in_proj</c>, <c>transformer.approximator.layers.{i}.in_layer|out_layer</c>,
/// <c>transformer.approximator.norms.{i}.weight</c>, <c>transformer.approximator.out_proj</c>; the T5 encoder under <c>text_encoder.</c> in
/// the encoder vocabulary of <see cref="EncoderDecoderAdapter"/>; the VAE as on <see cref="LatentDiffusionAdapter"/>.
/// Metadata: <c>chroma.transformer.*</c> (<see cref="FluxHyperParameters"/>), <c>chroma.vae.*</c>, <c>chroma.text_encoder.*</c> (the
/// encoder-decoder keys of the T5 encoder plus <c>prompt_length</c>), the T5 tokenizer under the canonical <c>tokenizer.*</c> keys,
/// <c>chroma.scheduler.*</c> (see <see cref="FlowSchedulerComponent"/>), <c>chroma.default_steps</c>, <c>chroma.default_guidance</c>.
/// A transformer-only source (as the GGUF files are distributed) gets the T5 encoder and the VAE as auxiliary sources
/// (<see cref="ChromaCompanionAdapter"/>).
/// </remarks>
public sealed class ChromaAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "chroma";

    private const string Prefix = Architecture + ".";
    private const string TextEncoderPrefix = "text_encoder.";
    private const string TransformerPrefix = "transformer.";
    private const string VaePrefix = "vae.";

    /// <inheritdoc/>
    public string Name => "Chroma (MMDiT with modulation approximator + VAE + T5)";

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
        var t5 = T5TextHyperParameters.Read(src, Prefix, "text_encoder");
        int downscale = vae.Downscale;
        long multiple = downscale * transformer.PatchSize;
        long height = options.Dimensions.GetValueOrDefault(DiffusionGraphs.Height, transformer.SampleSize);
        long width = options.Dimensions.GetValueOrDefault(DiffusionGraphs.Width, transformer.SampleSize);
        if (height <= 0 || width <= 0 || height % multiple != 0 || width % multiple != 0)
        {
            throw new ArgumentException($"Image size {width}x{height} must be a positive multiple of {multiple} pixels for {src.Get(CanonicalKeys.Name, src.Architecture)}.");
        }

        long latentHeight = height / downscale;
        long latentWidth = width / downscale;
        var graphs = new Dictionary<string, Graph.ComputeGraph>
        {
            [DiffusionGraphs.TextEncode] = T5TextGraphBuilder.Build(src.Weight, t5, TextEncoderPrefix, DiffusionGraphs.TextEncode, DiffusionGraphs.TokenMask),
            [DiffusionGraphs.Denoise] = FluxTransformerGraphBuilder.Build(src.Weight, transformer, latentHeight, latentWidth, t5.ContextLength),
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
            Description = Describe(src, transformer, vae, t5, schedulerAttachment),
        };
    }

    /// <summary>Fails with the list of missing parts when a transformer-only source comes without its companions.</summary>
    private static void RequireComponents(ModelSource src)
    {
        var missing = new List<string>();
        if (!src.Has(Prefix + "text_encoder.hidden_size"))
        {
            missing.Add("text-encoder (T5-XXL)");
        }

        if (!src.Has(Prefix + "vae.block_channels"))
        {
            missing.Add("vae");
        }

        if (missing.Count > 0)
        {
            throw new NotSupportedException($"'{src.Path}' holds only the Chroma transformer; it needs its companion files, given as auxiliary sources by role "
                + $"(or placed next to it): {string.Join(", ", missing)}.");
        }
    }

    private static ModelDescription Describe(ModelSource src, FluxHyperParameters transformer, AutoencoderHyperParameters vae, T5TextHyperParameters t5,
        bool hasSchedulerAttachment)
    {
        var vaeConfig = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.LatentChannels, vae.LatentChannels)
            .Add(DiffusionConfigKeys.Downscale, vae.Downscale);
        if (vae.LatentRgbFactors.Length > 0)
        {
            vaeConfig.Add(DiffusionConfigKeys.LatentRgbFactors, vae.LatentRgbFactors);
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
                    Config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, t5.ContextLength).Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, t5.ContextLength).Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Denoiser,
                    Graphs = [DiffusionGraphs.Denoise],
                    WeightPrefixes = [TransformerPrefix],
                    Config = new ComponentConfigBuilder()
                        .Add(DiffusionConfigKeys.LatentChannels, transformer.LatentChannels)
                        .Add(DiffusionConfigKeys.PatchSize, transformer.PatchSize)
                        .Add(DiffusionConfigKeys.ContextMaskPadding, 1)
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
                FlowSchedulerComponent.Create(src, Prefix, hasSchedulerAttachment),
            ],
            Config = new ComponentConfigBuilder()
                .Add(DiffusionConfigKeys.Width, transformer.SampleSize)
                .Add(DiffusionConfigKeys.Height, transformer.SampleSize)
                .Add(DiffusionConfigKeys.Steps, src.Get(Prefix + "default_steps", 35))
                .Add(DiffusionConfigKeys.GuidanceScale, src.Get(Prefix + "default_guidance", 5.0))
                .Build(),
        };
    }
}
