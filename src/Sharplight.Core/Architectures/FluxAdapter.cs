using System.Text;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds FLUX-style rectified-flow image generators (architecture <c>flux</c>: FLUX.1-schnell, FLUX.1-dev and their fine-tunes) as an
/// <see cref="ModelKinds.ImageGeneration"/> model: a CLIP text encoder for the pooled embedding, a T5 encoder for the per-token
/// conditioning, the MMDiT denoiser (<see cref="FluxTransformerGraphBuilder"/>) and a 16-channel KL autoencoder, as the graphs of
/// <see cref="DiffusionGraphs"/>, plus both tokenizers and a flow-matching scheduler component. The built-in latent-diffusion pipeline
/// runs it (one denoiser row per run; guidance-distilled models take the guidance scale as a denoiser input).
/// </summary>
/// <remarks>
/// Canonical vocabulary: the transformer as documented on <see cref="FluxTransformerGraphBuilder"/>; the VAE and the CLIP encoder
/// (<c>text_encoder.</c>) as on <see cref="LatentDiffusionAdapter"/>; the T5 encoder under <c>text_encoder_2.</c> in the encoder
/// vocabulary of <see cref="EncoderDecoderAdapter"/> (<c>token_embedding.weight</c>, <c>encoder.layers.{i}.*</c>,
/// <c>encoder.attention.relative_bias.weight</c>, <c>encoder.output_norm.weight</c>).
/// Metadata: <c>flux.transformer.*</c> (<see cref="FluxHyperParameters"/>), <c>flux.vae.*</c>, <c>flux.text_encoder.*</c> (CLIP),
/// <c>flux.text_encoder_2.*</c> (the encoder-decoder keys of the T5 encoder plus <c>prompt_length</c>), the CLIP tokenizer under the
/// canonical <c>tokenizer.*</c> keys and the T5 tokenizer under <c>flux.tokenizer_2.tokenizer.*</c>, <c>flux.scheduler.*</c>
/// (<c>train_steps</c>, <c>shift</c>, <c>dynamic_shift</c>, <c>base_shift</c>, <c>max_shift</c>, <c>base_sequence_length</c>,
/// <c>max_sequence_length</c>, <c>sampler</c>, <c>source_config</c>), <c>flux.default_steps</c> and <c>flux.default_guidance</c>.
/// <para>
/// A transformer-only source (as city96 / stable-diffusion.cpp GGUF files are distributed) lacks the text encoders and the VAE: load
/// them as auxiliary sources (<see cref="FluxCompanionAdapter"/>).
/// </para>
/// </remarks>
public sealed class FluxAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "flux";

    private const string Prefix = Architecture + ".";
    private const string TextEncoderPrefix = "text_encoder.";
    private const string SecondTextEncoderPrefix = "text_encoder_2.";
    private const string TransformerPrefix = "transformer.";
    private const string VaePrefix = "vae.";
    private const string SecondTokenizerMetadata = Prefix + "tokenizer_2.";
    private const string SchedulerAttachment = "scheduler/source-config.json";

    /// <inheritdoc/>
    public string Name => "FLUX (MMDiT + VAE + CLIP + T5)";

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
        var clip = ClipTextHyperParameters.Read(src, Prefix, "text_encoder");
        var t5 = T5TextHyperParameters.Read(src, Prefix, "text_encoder_2");
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
            [DiffusionGraphs.TextEncode] = ClipTextGraphBuilder.Build(src.Weight, clip, TextEncoderPrefix, pooled: true, hidden: false),
            [DiffusionGraphs.TextEncode2] = T5TextGraphBuilder.Build(src.Weight, t5, SecondTextEncoderPrefix),
            [DiffusionGraphs.Denoise] = FluxTransformerGraphBuilder.Build(src.Weight, transformer, latentHeight, latentWidth, t5.ContextLength),
            [DiffusionGraphs.VaeDecode] = AutoencoderGraphBuilder.BuildDecoder(src.Weight, vae, latentHeight, latentWidth),
            [DiffusionGraphs.VaeEncode] = AutoencoderGraphBuilder.BuildEncoder(src.Weight, vae, height, width),
        };

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
            Description = Describe(src, transformer, vae, clip, t5, attachments.Count > 0),
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

        if (!src.Has(Prefix + "text_encoder_2.hidden_size"))
        {
            missing.Add("text-encoder-2 (T5-XXL)");
        }

        if (!src.Has(Prefix + "vae.block_channels"))
        {
            missing.Add("vae");
        }

        if (missing.Count > 0)
        {
            throw new NotSupportedException($"'{src.Path}' holds only the FLUX transformer; it needs its companion files, given as auxiliary sources by role "
                + $"(or placed next to it): {string.Join(", ", missing)}.");
        }
    }

    private static ModelDescription Describe(ModelSource src, FluxHyperParameters transformer, AutoencoderHyperParameters vae, ClipTextHyperParameters clip,
        T5TextHyperParameters t5, bool hasSchedulerAttachment)
    {
        string s = Prefix + "scheduler.";
        var scheduler = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.TrainSteps, src.Get(s + "train_steps", 1000))
            .Add(DiffusionConfigKeys.PredictionType, "flow")
            .Add(DiffusionConfigKeys.Shift, src.Get(s + "shift", 1f))
            .Add(DiffusionConfigKeys.DynamicShift, src.Get(s + "dynamic_shift", false))
            .Add(DiffusionConfigKeys.BaseShift, src.Get(s + "base_shift", 0.5))
            .Add(DiffusionConfigKeys.MaxShift, src.Get(s + "max_shift", 1.15))
            .Add(DiffusionConfigKeys.BaseSequenceLength, src.Get(s + "base_sequence_length", 256))
            .Add(DiffusionConfigKeys.MaxSequenceLength, src.Get(s + "max_sequence_length", 4096))
            .Add(DiffusionConfigKeys.Sampler, src.Get(s + "sampler", Samplers.Euler));
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
                    Config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, clip.ContextLength).Build(),
                },
                new ModelComponent
                {
                    Role = DiffusionGraphs.SecondTextEncoderRole,
                    Graphs = [DiffusionGraphs.TextEncode2],
                    WeightPrefixes = [SecondTextEncoderPrefix],
                    Config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, t5.ContextLength).Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, clip.ContextLength).Build(),
                },
                new ModelComponent
                {
                    Role = DiffusionGraphs.SecondTokenizerRole,
                    MetadataPrefix = SecondTokenizerMetadata,
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
                new ModelComponent
                {
                    Role = ModelComponentRoles.Scheduler,
                    Config = scheduler.Build(),
                    Attachments = hasSchedulerAttachment
                        ? new Dictionary<string, string> { ["source-config"] = SchedulerAttachment }
                        : new Dictionary<string, string>(),
                },
            ],
            Config = new ComponentConfigBuilder()
                .Add(DiffusionConfigKeys.Width, transformer.SampleSize)
                .Add(DiffusionConfigKeys.Height, transformer.SampleSize)
                .Add(DiffusionConfigKeys.Steps, src.Get(Prefix + "default_steps", transformer.GuidanceEmbed ? 28 : 4))
                .Add(DiffusionConfigKeys.GuidanceScale, src.Get(Prefix + "default_guidance", transformer.GuidanceEmbed ? 3.5 : 0.0))
                .Add(DiffusionConfigKeys.ClassifierFreeGuidance, false)
                .Build(),
        };
    }
}
