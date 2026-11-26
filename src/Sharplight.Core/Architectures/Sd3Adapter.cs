using Sharplight.ImageGeneration;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Builds SD3-style rectified-flow image generators (architecture <c>sd3</c>: Stable Diffusion 3 Medium, Stable Diffusion 3.5 Medium /
/// Large and their fine-tunes) as an <see cref="ModelKinds.ImageGeneration"/> model: two CLIP text encoders (CLIP-L and OpenCLIP bigG,
/// penultimate hidden states joined per token, projected pooled embeddings concatenated), a T5 encoder whose hidden states follow the
/// CLIP ones along the sequence, the MMDiT denoiser (<see cref="Sd3TransformerGraphBuilder"/>) and a 16-channel KL autoencoder, as the
/// graphs of <see cref="DiffusionGraphs"/>, plus the CLIP and T5 tokenizers and a flow-matching scheduler component. The built-in
/// latent-diffusion pipeline runs it with classifier-free guidance (one denoiser row per run).
/// </summary>
/// <remarks>
/// Canonical vocabulary: the transformer as documented on <see cref="Sd3TransformerGraphBuilder"/>; the VAE and the CLIP encoders
/// (<c>text_encoder.</c>, <c>text_encoder_2.</c>, each with its <c>projection.weight</c>) as on <see cref="LatentDiffusionAdapter"/>; the
/// T5 encoder under <c>text_encoder_3.</c> in the encoder vocabulary of <see cref="EncoderDecoderAdapter"/>.
/// Metadata: <c>sd3.transformer.*</c> (<see cref="Sd3HyperParameters"/>), <c>sd3.vae.*</c>, <c>sd3.text_encoder.*</c> and
/// <c>sd3.text_encoder_2.*</c> (CLIP; <c>sd3.text_encoder_2.pad_id</c> the second encoder's padding id), <c>sd3.text_encoder_3.*</c> (the
/// encoder-decoder keys of the T5 encoder plus <c>prompt_length</c>), the CLIP tokenizer under the canonical <c>tokenizer.*</c> keys and
/// the T5 tokenizer under <c>sd3.tokenizer_3.tokenizer.*</c>, <c>sd3.scheduler.*</c> (see <see cref="FlowSchedulerComponent"/>),
/// <c>sd3.default_steps</c> and <c>sd3.default_guidance</c>.
/// <para>
/// A transformer-only source (as the GGUF files are distributed) lacks the text encoders and the VAE: load them as auxiliary sources
/// (<see cref="Sd3CompanionAdapter"/>).
/// </para>
/// </remarks>
public sealed class Sd3Adapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "sd3";

    private const string Prefix = Architecture + ".";
    private const string TextEncoderPrefix = "text_encoder.";
    private const string SecondTextEncoderPrefix = "text_encoder_2.";
    private const string ThirdTextEncoderPrefix = "text_encoder_3.";
    private const string TransformerPrefix = "transformer.";
    private const string VaePrefix = "vae.";
    private const string ThirdTokenizerMetadata = Prefix + "tokenizer_3.";

    /// <inheritdoc/>
    public string Name => "Stable Diffusion 3 (MMDiT + VAE + CLIP-L + CLIP-G + T5)";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures { get; } = [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        ArgumentNullException.ThrowIfNull(sources);
        ArgumentNullException.ThrowIfNull(options);
        var src = DiffusionAddOns.Merge(sources, acceptsControlNet: false);
        RequireComponents(src);
        var transformer = Sd3HyperParameters.Read(src, Prefix);
        var vae = AutoencoderHyperParameters.Read(src, Prefix);
        var clip = ClipTextHyperParameters.Read(src, Prefix, "text_encoder");
        var clip2 = ClipTextHyperParameters.Read(src, Prefix, "text_encoder_2");
        var t5 = T5TextHyperParameters.Read(src, Prefix, "text_encoder_3");
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
        int textRows = clip.ContextLength + t5.ContextLength;
        var graphs = new Dictionary<string, Graph.ComputeGraph>
        {
            [DiffusionGraphs.TextEncode] = ClipTextGraphBuilder.Build(src.Weight, clip, TextEncoderPrefix, pooled: true),
            [DiffusionGraphs.TextEncode2] = ClipTextGraphBuilder.Build(src.Weight, clip2, SecondTextEncoderPrefix, DiffusionGraphs.TextEncode2, pooled: true),
            [DiffusionGraphs.TextEncode3] = T5TextGraphBuilder.Build(src.Weight, t5, ThirdTextEncoderPrefix, DiffusionGraphs.TextEncode3),
            [DiffusionGraphs.Denoise] = Sd3TransformerGraphBuilder.Build(src.Weight, transformer, latentHeight, latentWidth, textRows),
            [DiffusionGraphs.VaeDecode] = AutoencoderGraphBuilder.BuildDecoder(src.Weight, vae, latentHeight, latentWidth),
            [DiffusionGraphs.VaeEncode] = AutoencoderGraphBuilder.BuildEncoder(src.Weight, vae, height, width),
        };
        DiffusionAddOns.AddTinyAutoencoder(src, graphs, vae.LatentChannels, downscale, latentHeight, latentWidth);

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
            Description = DiffusionAddOns.WithAddOns(Describe(src, transformer, vae, clip, t5, schedulerAttachment), graphs),
        };
    }

    /// <summary>Fails with the list of missing parts when a transformer-only source comes without its companions.</summary>
    private static void RequireComponents(ModelSource src)
    {
        var missing = new List<string>();
        if (!src.Has(Prefix + "text_encoder.layer_count"))
        {
            missing.Add("text-encoder (CLIP-L)");
        }

        if (!src.Has(Prefix + "text_encoder_2.layer_count"))
        {
            missing.Add("text-encoder-2 (CLIP-G)");
        }

        if (!src.Has(Prefix + "text_encoder_3.hidden_size"))
        {
            missing.Add("text-encoder-3 (T5-XXL)");
        }

        if (!src.Has(Prefix + "vae.block_channels"))
        {
            missing.Add("vae");
        }

        if (missing.Count > 0)
        {
            throw new NotSupportedException($"'{src.Path}' holds only the SD3 transformer; it needs its companion files, given as auxiliary sources by role "
                + $"(or placed next to it): {string.Join(", ", missing)}.");
        }
    }

    private static ModelDescription Describe(ModelSource src, Sd3HyperParameters transformer, AutoencoderHyperParameters vae, ClipTextHyperParameters clip,
        T5TextHyperParameters t5, bool hasSchedulerAttachment)
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
                    Graphs = [DiffusionGraphs.TextEncode, DiffusionGraphs.TextEncode2],
                    WeightPrefixes = [TextEncoderPrefix, SecondTextEncoderPrefix],
                    Config = new ComponentConfigBuilder()
                        .Add(DiffusionConfigKeys.ContextLength, clip.ContextLength)
                        .Add(DiffusionConfigKeys.SecondPadId, src.Get(Prefix + "text_encoder_2.pad_id", -1))
                        .Build(),
                },
                new ModelComponent
                {
                    Role = DiffusionGraphs.ThirdTextEncoderRole,
                    Graphs = [DiffusionGraphs.TextEncode3],
                    WeightPrefixes = [ThirdTextEncoderPrefix],
                    Config = new ComponentConfigBuilder()
                        .Add(DiffusionConfigKeys.ContextLength, t5.ContextLength)
                        .Add(DiffusionConfigKeys.JoinSequence, true)
                        .Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, clip.ContextLength).Build(),
                },
                new ModelComponent
                {
                    Role = DiffusionGraphs.ThirdTokenizerRole,
                    MetadataPrefix = ThirdTokenizerMetadata,
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
                .Add(DiffusionConfigKeys.Steps, src.Get(Prefix + "default_steps", 28))
                .Add(DiffusionConfigKeys.GuidanceScale, src.Get(Prefix + "default_guidance", 7.0))
                .Build(),
        };
    }
}
