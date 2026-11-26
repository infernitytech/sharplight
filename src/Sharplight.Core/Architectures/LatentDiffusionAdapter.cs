using System.Text;
using System.Text.Json;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds latent-diffusion image generators (architecture <c>latent_diffusion</c>: Stable Diffusion 1.x / 2.x / XL layout) as an
/// <see cref="ModelKinds.ImageGeneration"/> model: a CLIP text encoder, a cross-attention UNet denoiser and a KL autoencoder, as the
/// graphs of <see cref="DiffusionGraphs"/>, plus a tokenizer and a configuration-only scheduler component. The built-in
/// latent-diffusion pipeline of the task layer runs it.
/// </summary>
/// <remarks>
/// Canonical vocabulary (weights; every <c>.weight</c> has its <c>.bias</c> where the network has one):
/// <list type="bullet">
/// <item>UNet: <c>unet.time_embedding.linear1|linear2</c>, <c>unet.input_conv</c>, <c>unet.down.{l}.res.{i}.{norm1,conv1,time_proj,norm2,conv2,shortcut}</c>,
/// <c>unet.down.{l}.attn.{i}.{norm,proj_in,proj_out}</c> with <c>.blocks.{b}.{norm1,self.{q,k,v,o},norm2,cross.{q,k,v,o},norm3,ff.proj,ff.out}</c>
/// (<c>ff.proj</c> is the GEGLU projection, value half first), <c>unet.down.{l}.downsample</c>, <c>unet.mid.res.{0,1}</c>,
/// <c>unet.mid.attn.0</c>, <c>unet.up.{l}.res.{i}</c>, <c>unet.up.{l}.attn.{i}</c>, <c>unet.up.{l}.upsample</c> (up levels coarsest
/// first), <c>unet.output_norm</c>, <c>unet.output_conv</c>. Projections are matrices [out, in]; convolutions [out, in, k, k].</item>
/// <item>VAE: <c>vae.{encoder|decoder}.input_conv</c>, <c>vae.encoder.down.{l}.res.{i}</c>, <c>vae.encoder.down.{l}.downsample</c>,
/// <c>vae.decoder.up.{l}.res.{i}</c> (coarsest first), <c>vae.decoder.up.{l}.upsample</c>, <c>vae.{encoder|decoder}.mid.res.{0,1}</c>,
/// <c>vae.{encoder|decoder}.mid.attn.0.{norm,q,k,v,o}</c>, <c>vae.{encoder|decoder}.output_norm|output_conv</c>, <c>vae.quant_conv</c>,
/// <c>vae.post_quant_conv</c>.</item>
/// <item>Text encoder: <c>text_encoder.token_embedding</c>, <c>text_encoder.position_embedding</c> (f32),
/// <c>text_encoder.layers.{i}.{norm1,attn.{q,k,v,o},norm2,mlp.up,mlp.down}</c>, <c>text_encoder.final_norm</c>; SDXL's second
/// encoder the same under <c>text_encoder_2.</c> plus <c>text_encoder_2.projection.weight</c> (pooled embedding).</item>
/// <item>SDXL added conditioning: <c>unet.class_embedding.linear1|linear2</c>.</item>
/// </list>
/// Metadata: <c>latent_diffusion.unet.*</c> (<c>in_channels</c>, <c>out_channels</c>, <c>block_channels</c>, <c>layers_per_block</c>,
/// <c>down_transformer_depths</c>, <c>up_transformer_depths</c>, <c>head_counts</c>, <c>mid_transformer_depth</c>, <c>context_dim</c>,
/// <c>sample_size</c>, <c>norm_groups</c>, <c>norm_eps</c>, <c>flip_sin_to_cos</c>, <c>freq_shift</c>, <c>class_embedding_input</c>,
/// <c>addition_time_dim</c>), <c>latent_diffusion.vae.*</c>
/// (<c>latent_channels</c>, <c>block_channels</c>, <c>layers_per_block</c>, <c>norm_groups</c>, <c>scaling_factor</c>,
/// <c>shift_factor</c>, <c>quant_convolutions</c>, <c>latent_rgb_factors</c>), <c>latent_diffusion.text_encoder.*</c>
/// (<c>layer_count</c>, <c>hidden_size</c>, <c>head_count</c>, <c>activation</c>, <c>context_length</c>, <c>norm_eps</c>, <c>clip_skip</c>,
/// <c>final_norm</c>), optionally <c>latent_diffusion.text_encoder_2.*</c> (the same plus <c>pad_id</c>),
/// <c>latent_diffusion.zero_empty_negative</c>, <c>latent_diffusion.scheduler.*</c> (<c>train_steps</c>, <c>beta_start</c>, <c>beta_end</c>, <c>beta_schedule</c>,
/// <c>prediction_type</c>, <c>timestep_spacing</c>, <c>steps_offset</c>, <c>set_alpha_to_one</c>, <c>zero_terminal_snr</c>,
/// <c>sampler</c>, <c>source_config</c>, <c>default_steps</c>, <c>default_guidance</c>) and the canonical tokenizer keys (<c>tokenizer.model = "clip"</c>).
/// <para>
/// The graphs are built for the native image size, or for <see cref="DiffusionGraphs.Height"/> / <see cref="DiffusionGraphs.Width"/>
/// from <see cref="AdapterOptions.Dimensions"/> (multiples of the VAE downscale; the UNet needs multiples of 8 latents).
/// </para>
/// </remarks>
public sealed class LatentDiffusionAdapter : IModelAdapter
{
    private const string Prefix = "latent_diffusion.";
    private const string TextEncoderPrefix = "text_encoder.";
    private const string SecondTextEncoderPrefix = "text_encoder_2.";
    private const string SchedulerAttachment = "scheduler/source-config.json";

    /// <inheritdoc/>
    public string Name => "Latent diffusion (UNet + VAE + CLIP)";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures { get; } = ["latent_diffusion"];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        ArgumentNullException.ThrowIfNull(sources);
        ArgumentNullException.ThrowIfNull(options);
        var src = DiffusionAddOns.Merge(sources, acceptsControlNet: true);
        var unet = UNetHyperParameters.Read(src, Prefix);
        var vae = AutoencoderHyperParameters.Read(src, Prefix);
        var text = ClipTextHyperParameters.Read(src, Prefix, "text_encoder");
        var secondText = src.Has(Prefix + "text_encoder_2.layer_count") ? ClipTextHyperParameters.Read(src, Prefix, "text_encoder_2") : null;
        // The pooled embedding comes from the second encoder's projection (SDXL), or the only encoder's (the SDXL refiner's OpenCLIP bigG).
        var projection = src.Weight(SecondTextEncoderPrefix + "projection.weight") ?? src.Weight(TextEncoderPrefix + "projection.weight");
        unet = unet.WithPooledWidth(projection is null ? 0 : (int)projection.Shape[0].Fixed);
        int downscale = vae.Downscale;
        long height = options.Dimensions.GetValueOrDefault(DiffusionGraphs.Height, unet.SampleSize * downscale);
        long width = options.Dimensions.GetValueOrDefault(DiffusionGraphs.Width, unet.SampleSize * downscale);
        long unetMultiple = downscale << (unet.BlockChannels.Length - 1);
        if (height <= 0 || width <= 0 || height % unetMultiple != 0 || width % unetMultiple != 0)
        {
            throw new ArgumentException($"Image size {width}x{height} must be a positive multiple of {unetMultiple} pixels for {src.Get(CanonicalKeys.Name, src.Architecture)}.");
        }

        long latentHeight = height / downscale;
        long latentWidth = width / downscale;
        var graphs = new Dictionary<string, Graph.ComputeGraph>
        {
            [DiffusionGraphs.TextEncode] = ClipTextGraphBuilder.Build(src.Weight, text, TextEncoderPrefix,
                pooled: secondText is null && src.Weight(TextEncoderPrefix + "projection.weight") is not null),
            [DiffusionGraphs.Denoise] = UNetGraphBuilder.Build(src.Weight, unet, latentHeight, latentWidth, text.ContextLength),
            [DiffusionGraphs.VaeDecode] = AutoencoderGraphBuilder.BuildDecoder(src.Weight, vae, latentHeight, latentWidth),
            [DiffusionGraphs.VaeEncode] = AutoencoderGraphBuilder.BuildEncoder(src.Weight, vae, height, width),
        };
        DiffusionAddOns.AddTinyAutoencoder(src, graphs, vae.LatentChannels, downscale, latentHeight, latentWidth);
        if (ControlNetHyperParameters.IsPresent(src))
        {
            var control = ControlNetHyperParameters.Read(src);
            if (control.UNet.ContextDim != unet.ContextDim || control.UNet.BlockChannels.Length != unet.BlockChannels.Length
                || control.UNet.ClassEmbeddingInput != unet.ClassEmbeddingInput)
            {
                throw new NotSupportedException($"The ControlNet ({control.UNet.BlockChannels.Length} levels, {control.UNet.ContextDim}-wide text conditioning) "
                    + $"is not for {src.Get(CanonicalKeys.Name, src.Architecture)} ({unet.BlockChannels.Length} levels, {unet.ContextDim}-wide): SD 1.x ControlNets "
                    + "condition SD 1.x UNets, SDXL ControlNets SDXL UNets.");
            }

            graphs[DiffusionGraphs.DenoiseControl] = UNetGraphBuilder.Build(src.Weight, unet, latentHeight, latentWidth, text.ContextLength, control);
        }

        if (secondText is not null)
        {
            graphs[DiffusionGraphs.TextEncode2] = ClipTextGraphBuilder.Build(src.Weight, secondText, SecondTextEncoderPrefix, DiffusionGraphs.TextEncode2,
                pooled: src.Weight(SecondTextEncoderPrefix + "projection.weight") is not null);
        }

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
            Description = DiffusionAddOns.WithAddOns(Describe(src, unet, vae, text, secondText is not null, attachments.Count > 0), graphs),
        };
    }

    private static IReadOnlyDictionary<string, JsonElement> DenoiserConfig(UNetHyperParameters unet, ModelSource src)
    {
        var config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.LatentChannels, unet.InChannels - unet.ConditionChannels);
        if (unet.ConditionChannels > 0)
        {
            // 4 extra channels: InstructPix2Pix's image latents; 5: the inpainting UNets' mask and masked-image latents.
            config.Add(DiffusionConfigKeys.ConditionMode, src.Get(Prefix + "unet.condition_mode", unet.ConditionChannels == 4 ? DiffusionConfigKeys.ConditionImage : DiffusionConfigKeys.ConditionInpaint));
        }

        return config.Build();
    }

    private static ModelDescription Describe(ModelSource src, UNetHyperParameters unet, AutoencoderHyperParameters vae, ClipTextHyperParameters text,
        bool secondEncoder, bool hasSchedulerAttachment)
    {
        var textConfig = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.ContextLength, text.ContextLength)
            .Add(DiffusionConfigKeys.ZeroEmptyNegative, src.Get(Prefix + "zero_empty_negative", false));
        if (secondEncoder)
        {
            textConfig.Add(DiffusionConfigKeys.SecondPadId, src.Get(Prefix + "text_encoder_2.pad_id", -1));
        }

        string s = Prefix + "scheduler.";
        var scheduler = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.TrainSteps, src.Get(s + "train_steps", 1000))
            .Add(DiffusionConfigKeys.BetaStart, src.Get(s + "beta_start", 0.00085f))
            .Add(DiffusionConfigKeys.BetaEnd, src.Get(s + "beta_end", 0.012f))
            .Add(DiffusionConfigKeys.BetaSchedule, src.Get(s + "beta_schedule", "scaled_linear"))
            .Add(DiffusionConfigKeys.PredictionType, src.Get(s + "prediction_type", "epsilon"))
            .Add(DiffusionConfigKeys.StepsOffset, src.Get(s + "steps_offset", 0))
            .Add(DiffusionConfigKeys.ZeroTerminalSnr, src.Get(s + "zero_terminal_snr", false))
            .Add(DiffusionConfigKeys.Sampler, src.Get(s + "sampler", "euler"));
        if (src.Get<string?>(s + "timestep_spacing", null) is { } spacing)
        {
            scheduler.Add(DiffusionConfigKeys.TimestepSpacing, spacing);
        }

        if (src.Has(s + "set_alpha_to_one"))
        {
            scheduler.Add(DiffusionConfigKeys.SetAlphaToOne, src.Get<bool>(s + "set_alpha_to_one"));
        }

        if (src.Has(s + "clip_sample"))
        {
            scheduler.Add(DiffusionConfigKeys.ClipSample, src.Get<bool>(s + "clip_sample"));
        }

        var vaeConfig = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.LatentChannels, vae.LatentChannels)
            .Add(DiffusionConfigKeys.Downscale, vae.Downscale)
            .Add(DiffusionConfigKeys.ScalingFactor, vae.ScalingFactor);
        if (vae.LatentRgbFactors.Length > 0)
        {
            vaeConfig.Add(DiffusionConfigKeys.LatentRgbFactors, vae.LatentRgbFactors);
        }

        long batch1 = 1;
        // A guided step evaluates two rows (unconditional and conditional); with image guidance (InstructPix2Pix) three.
        long denoiserRows = unet.ConditionChannels > 0
            && src.Get(Prefix + "unet.condition_mode", unet.ConditionChannels == 4 ? DiffusionConfigKeys.ConditionImage : DiffusionConfigKeys.ConditionInpaint)
                == DiffusionConfigKeys.ConditionImage ? 3 : 2;
        int nativeSize = unet.SampleSize * vae.Downscale;
        return new ModelDescription
        {
            Kind = ModelKinds.ImageGeneration,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.TextEncoder,
                    Graphs = secondEncoder ? [DiffusionGraphs.TextEncode, DiffusionGraphs.TextEncode2] : [DiffusionGraphs.TextEncode],
                    WeightPrefixes = secondEncoder ? [TextEncoderPrefix, SecondTextEncoderPrefix] : [TextEncoderPrefix],
                    Config = textConfig.Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Tokenizer,
                    Config = new ComponentConfigBuilder().Add(DiffusionConfigKeys.ContextLength, text.ContextLength).Build(),
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Denoiser,
                    Graphs = [DiffusionGraphs.Denoise],
                    WeightPrefixes = ["unet."],
                    Config = DenoiserConfig(unet, src),
                    SymbolBounds = new Dictionary<string, long> { [DiffusionGraphs.BatchSymbol] = denoiserRows },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.Vae,
                    Graphs = [DiffusionGraphs.VaeDecode, DiffusionGraphs.VaeEncode],
                    WeightPrefixes = ["vae."],
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
            Config = ModelConfig(src, s, nativeSize, unet),
        };
    }

    private static IReadOnlyDictionary<string, JsonElement> ModelConfig(ModelSource src, string s, int nativeSize, UNetHyperParameters unet)
    {
        var config = new ComponentConfigBuilder()
            .Add(DiffusionConfigKeys.Width, nativeSize)
            .Add(DiffusionConfigKeys.Height, nativeSize)
            .Add(DiffusionConfigKeys.Steps, src.Get(s + "default_steps", 25))
            .Add(DiffusionConfigKeys.GuidanceScale, src.Get(s + "default_guidance", 7.5f));
        if (unet.ConditionChannels == 4)
        {
            config.Add(DiffusionConfigKeys.ImageGuidanceScale, src.Get(s + "default_image_guidance", 1.5f));
        }

        return config.Build();
    }
}
