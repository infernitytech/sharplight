using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The configuration of a FLUX transformer and its pipeline defaults, written as the canonical <c>flux.transformer.*</c>,
/// <c>flux.scheduler.*</c> and default keys. The shapes come from the canonical weights (every format stores the same tensors);
/// the guidance embedding tells the guidance-distilled models (FLUX.1-dev: dynamic timestep shift, 28 steps, guidance 3.5, T5
/// prompts of 512 tokens) from the timestep-distilled ones (FLUX.1-schnell: no shift, 4 steps, no guidance, 256 tokens).
/// </summary>
internal sealed record FluxConfig
{
    /// <summary>Canonical architecture id.</summary>
    public const string Architecture = "flux";

    /// <summary>Prefix of the architecture's metadata keys.</summary>
    public const string Prefix = Architecture + ".";

    /// <summary>The FLUX VAE's latent scaling.</summary>
    public const float VaeScalingFactor = 0.3611f;

    /// <summary>The FLUX VAE's latent shift.</summary>
    public const float VaeShiftFactor = 0.1159f;

    /// <summary>Latent-to-RGB preview factors of the FLUX latent space ([16, 3], row-major).</summary>
    public static readonly float[] RgbFactors =
    [
        -0.0346f, 0.0244f, 0.0681f, 0.0034f, 0.0210f, 0.0687f, 0.0275f, -0.0668f, -0.0433f, -0.0174f, 0.0160f, 0.0617f,
        0.0859f, 0.0721f, 0.0329f, 0.0004f, 0.0383f, 0.0115f, 0.0405f, 0.0861f, 0.0915f, -0.0236f, -0.0185f, -0.0259f,
        -0.0245f, 0.0250f, 0.1180f, 0.1008f, 0.0755f, -0.0421f, -0.0515f, 0.0201f, 0.0011f, 0.0428f, -0.0012f, -0.0036f,
        0.0817f, 0.0765f, 0.0749f, -0.1264f, -0.0522f, -0.1103f, -0.0280f, -0.0881f, -0.0499f, -0.1262f, -0.0982f, -0.0778f,
    ];

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Dual-stream blocks.</summary>
    public required int DoubleLayers { get; init; }

    /// <summary>Single-stream blocks.</summary>
    public required int SingleLayers { get; init; }

    /// <summary>Text conditioning width.</summary>
    public required int ContextDim { get; init; }

    /// <summary>Pooled text embedding width.</summary>
    public required int PooledDim { get; init; }

    /// <summary>Patch input width (latent channels × patch area).</summary>
    public required int InChannels { get; init; }

    /// <summary>Whether the guidance scale is embedded.</summary>
    public required bool GuidanceEmbed { get; init; }

    /// <summary>RoPE dimensions per position axis.</summary>
    public int[] AxesDim { get; init; } = [16, 56, 56];

    /// <summary>Latent channels.</summary>
    public int LatentChannels { get; init; } = 16;

    /// <summary>Flow scheduler: static shift.</summary>
    public float Shift { get; init; } = 1f;

    /// <summary>Flow scheduler: resolution-dependent shift.</summary>
    public bool DynamicShift { get; init; }

    /// <summary>Flow scheduler: μ at the base sequence length.</summary>
    public double BaseShift { get; init; } = 0.5;

    /// <summary>Flow scheduler: μ at the maximum sequence length.</summary>
    public double MaxShift { get; init; } = 1.15;

    /// <summary>Flow scheduler: image tokens of the base shift.</summary>
    public int BaseSequenceLength { get; init; } = 256;

    /// <summary>Flow scheduler: image tokens of the maximum shift.</summary>
    public int MaxSequenceLength { get; init; } = 4096;

    /// <summary>The original scheduler configuration, or null.</summary>
    public string? SchedulerJson { get; init; }

    /// <summary>
    /// Whether the transformer is FLUX.1 Kontext (the FLUX.1-dev layout trained to edit a reference image whose tokens follow the image's at
    /// the first position index 1; recognized by the pipeline class or the file name): 28 steps with the embedded guidance 2.5.
    /// </summary>
    public bool Kontext { get; init; }

    /// <summary>T5 prompt length.</summary>
    public int PromptLength => GuidanceEmbed ? 512 : 256;

    /// <summary>Infers the configuration from canonical <c>transformer.*</c> weights, with the family's scheduler defaults.</summary>
    /// <param name="weights">The canonical weights.</param>
    /// <exception cref="InvalidDataException">The weights are not a FLUX transformer.</exception>
    public static FluxConfig FromWeights(IReadOnlyDictionary<string, WeightInfo> weights)
    {
        if (!weights.TryGetValue("transformer.img_in.weight", out var imageIn) || !weights.TryGetValue("transformer.double.0.img_attn.norm_q.weight", out var norm))
        {
            throw new InvalidDataException("The checkpoint has no FLUX transformer.");
        }

        int hidden = (int)imageIn.Shape[0].Fixed;
        int headDim = (int)norm.Shape[0].Fixed;
        bool guidance = weights.ContainsKey("transformer.guidance_in.in_layer.weight");
        return new FluxConfig
        {
            HiddenSize = hidden,
            HeadCount = hidden / headDim,
            DoubleLayers = FluxNames.CountBlocks(weights, "double"),
            SingleLayers = FluxNames.CountBlocks(weights, "single"),
            ContextDim = (int)weights["transformer.txt_in.weight"].Shape[1].Fixed,
            PooledDim = (int)weights["transformer.vector_in.in_layer.weight"].Shape[1].Fixed,
            InChannels = (int)imageIn.Shape[1].Fixed,
            GuidanceEmbed = guidance,
            DynamicShift = guidance,
            // FLUX.1 rotates 16 + 56 + 56 of its 128 head dimensions by (index, row, column); files without a configuration keep those proportions.
            AxesDim = [headDim / 8, headDim * 7 / 16, headDim * 7 / 16],
        };
    }

    /// <summary>This configuration with the scheduler settings and RoPE axes of a diffusers folder.</summary>
    /// <param name="schedulerJson">The <c>scheduler_config.json</c> text, or null.</param>
    /// <param name="transformerConfig">The <c>transformer/config.json</c>, or null.</param>
    public FluxConfig WithDiffusers(string? schedulerJson, JsonElement? transformerConfig)
    {
        var result = this;
        if (transformerConfig is { } config && config.TryGetProperty("axes_dims_rope", out _))
        {
            result = result with { AxesDim = config.Ints("axes_dims_rope") };
        }

        if (schedulerJson is null)
        {
            return result;
        }

        using var document = JsonDocument.Parse(schedulerJson);
        var s = document.RootElement;
        return result with
        {
            Shift = s.Float("shift", 1f),
            DynamicShift = s.Bool("use_dynamic_shifting", false),
            BaseShift = s.TryGetProperty("base_shift", out var b) ? b.GetDouble() : 0.5,
            MaxShift = s.TryGetProperty("max_shift", out var m) ? m.GetDouble() : 1.15,
            BaseSequenceLength = s.Int("base_image_seq_len", 256),
            MaxSequenceLength = s.Int("max_image_seq_len", 4096),
            SchedulerJson = schedulerJson,
        };
    }

    /// <summary>Writes the transformer, scheduler, family VAE and default keys.</summary>
    /// <param name="metadata">The canonical metadata.</param>
    public void WriteTo(IDictionary<string, object> metadata)
    {
        string t = Prefix + "transformer.";
        int patch = (int)Math.Round(Math.Sqrt(InChannels / (double)LatentChannels));
        metadata[t + "latent_channels"] = LatentChannels;
        metadata[t + "patch_size"] = patch;
        metadata[t + "hidden_size"] = HiddenSize;
        metadata[t + "head_count"] = HeadCount;
        metadata[t + "double_layers"] = DoubleLayers;
        metadata[t + "single_layers"] = SingleLayers;
        metadata[t + "context_dim"] = ContextDim;
        metadata[t + "pooled_dim"] = PooledDim;
        metadata[t + "guidance_embed"] = GuidanceEmbed;
        metadata[t + "axes_dim"] = AxesDim;
        metadata[t + "rope_theta"] = 10000f;
        metadata[t + "sample_size"] = 1024;
        if (Kontext)
        {
            metadata[t + "max_references"] = 1;
            metadata[t + "reference_index_base"] = 1;
            metadata[t + "reference_index_step"] = 0;
        }

        string s = Prefix + "scheduler.";
        metadata[s + "train_steps"] = 1000;
        metadata[s + "shift"] = Shift;
        metadata[s + "dynamic_shift"] = DynamicShift;
        metadata[s + "base_shift"] = BaseShift;
        metadata[s + "max_shift"] = MaxShift;
        metadata[s + "base_sequence_length"] = BaseSequenceLength;
        metadata[s + "max_sequence_length"] = MaxSequenceLength;
        metadata[s + "sampler"] = "euler";
        if (SchedulerJson is not null)
        {
            metadata[s + "source_config"] = SchedulerJson;
        }

        metadata[Prefix + "text_encoder_2.prompt_length"] = PromptLength;
        metadata[Prefix + "vae.scaling_factor"] = VaeScalingFactor;
        metadata[Prefix + "vae.shift_factor"] = VaeShiftFactor;
        metadata[Prefix + "vae.latent_rgb_factors"] = RgbFactors;
        metadata[Prefix + "default_steps"] = GuidanceEmbed ? 28 : 4;
        metadata[Prefix + "default_guidance"] = Kontext ? 2.5 : GuidanceEmbed ? 3.5 : 0.0;
    }
}
