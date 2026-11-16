using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The configuration of a Chroma transformer (FLUX.1-schnell layout with the modulation approximator) and its pipeline defaults,
/// written as the canonical <c>chroma.transformer.*</c>, <c>chroma.scheduler.*</c> and default keys, plus the family keys of its
/// companions (T5 prompts of 512 tokens, the FLUX VAE's latent scaling). The shapes come from the canonical weights.
/// </summary>
internal sealed record ChromaConfig
{
    /// <summary>Canonical architecture id.</summary>
    public const string Architecture = "chroma";

    /// <summary>Prefix of the architecture's metadata keys.</summary>
    public const string Prefix = Architecture + ".";

    /// <summary>T5 prompt length of the Chroma pipeline.</summary>
    public const int PromptLength = 512;

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

    /// <summary>Patch input width (latent channels × patch area).</summary>
    public required int InChannels { get; init; }

    /// <summary>Approximator layers.</summary>
    public required int ApproximatorLayers { get; init; }

    /// <summary>Approximator input width.</summary>
    public required int ApproximatorChannels { get; init; }

    /// <summary>RoPE dimensions per position axis.</summary>
    public int[] AxesDim { get; init; } = [16, 56, 56];

    /// <summary>Latent channels.</summary>
    public int LatentChannels { get; init; } = 16;

    /// <summary>Flow scheduler: resolution-dependent shift (Chroma1-HD's scheduler: off).</summary>
    public bool DynamicShift { get; init; }

    /// <summary>Flow scheduler: static shift (Chroma1-HD's scheduler: 3).</summary>
    public float Shift { get; init; } = 3f;

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

    /// <summary>Whether canonical <c>transformer.*</c> weights are a Chroma transformer (they have the approximator).</summary>
    /// <param name="weights">The canonical weights.</param>
    public static bool IsChroma(IReadOnlyDictionary<string, WeightInfo> weights) => weights.ContainsKey("transformer.approximator.in_proj.weight");

    /// <summary>Infers the configuration from canonical <c>transformer.*</c> weights, with the family's defaults.</summary>
    /// <param name="weights">The canonical weights.</param>
    /// <exception cref="InvalidDataException">The weights are not a Chroma transformer.</exception>
    public static ChromaConfig FromWeights(IReadOnlyDictionary<string, WeightInfo> weights)
    {
        if (!weights.TryGetValue("transformer.img_in.weight", out var imageIn) || !weights.TryGetValue("transformer.double.0.img_attn.norm_q.weight", out var norm)
            || !weights.TryGetValue("transformer.approximator.in_proj.weight", out var approximator))
        {
            throw new InvalidDataException("The checkpoint has no Chroma transformer.");
        }

        int hidden = (int)imageIn.Shape[0].Fixed;
        int headDim = (int)norm.Shape[0].Fixed;
        return new ChromaConfig
        {
            HiddenSize = hidden,
            HeadCount = hidden / headDim,
            DoubleLayers = FluxNames.CountBlocks(weights, "double"),
            SingleLayers = FluxNames.CountBlocks(weights, "single"),
            ContextDim = (int)weights["transformer.txt_in.weight"].Shape[1].Fixed,
            InChannels = (int)imageIn.Shape[1].Fixed,
            ApproximatorLayers = FluxNames.CountApproximatorLayers(weights),
            ApproximatorChannels = (int)approximator.Shape[1].Fixed,
            AxesDim = [headDim / 8, headDim * 7 / 16, headDim * 7 / 16],
        };
    }

    /// <summary>This configuration with the scheduler settings and RoPE axes of a diffusers folder.</summary>
    /// <param name="schedulerJson">The <c>scheduler_config.json</c> text, or null.</param>
    /// <param name="transformerConfig">The <c>transformer/config.json</c>, or null.</param>
    public ChromaConfig WithDiffusers(string? schedulerJson, JsonElement? transformerConfig)
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

    /// <summary>Writes the transformer, scheduler, family and default keys.</summary>
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
        metadata[t + "guidance_embed"] = false;
        metadata[t + "axes_dim"] = AxesDim;
        metadata[t + "rope_theta"] = 10000f;
        metadata[t + "sample_size"] = 1024;
        metadata[t + "approximator_layers"] = ApproximatorLayers;
        metadata[t + "approximator_channels"] = ApproximatorChannels;
        metadata[t + "masked_text"] = true;
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

        metadata[Prefix + "text_encoder.prompt_length"] = PromptLength;
        metadata[Prefix + "vae.scaling_factor"] = FluxConfig.VaeScalingFactor;
        metadata[Prefix + "vae.shift_factor"] = FluxConfig.VaeShiftFactor;
        metadata[Prefix + "vae.latent_rgb_factors"] = FluxConfig.RgbFactors;
        metadata[Prefix + "default_steps"] = 35;
        metadata[Prefix + "default_guidance"] = 5.0;
    }
}
