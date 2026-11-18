using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The configuration of a Z-Image transformer and its pipeline defaults, written as the canonical <c>zimage.transformer.*</c>,
/// <c>zimage.scheduler.*</c>, text-encoder and default keys (Z-Image-Turbo: shift 3, 9 steps, no guidance; the Qwen3 encoder's
/// second-to-last hidden states over the chat-templated prompt of at most 512 tokens; the FLUX VAE's latent scaling).
/// </summary>
internal sealed record ZImageConfig
{
    /// <summary>Canonical architecture id.</summary>
    public const string Architecture = "zimage";

    /// <summary>Prefix of the architecture's metadata keys.</summary>
    public const string Prefix = Architecture + ".";

    /// <summary>The Qwen3 chat template the pipeline puts the prompt into (one user turn, thinking enabled: no empty think block).</summary>
    public const string PromptTemplate = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";

    /// <summary>
    /// The undistilled model's default guidance scale. diffusers' Z-Image pipeline combines <c>pos + g · (pos − neg)</c> with its default
    /// <c>guidance_scale</c> 4, which is the usual <c>neg + (1 + g) · (pos − neg)</c> with scale 5 (as for Krea 2).
    /// </summary>
    public const double BaseGuidance = 5.0;

    /// <summary>Prompt tokens of the pipeline.</summary>
    public const int PromptLength = 512;

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Joint blocks.</summary>
    public required int Layers { get; init; }

    /// <summary>Refiner blocks per stream.</summary>
    public required int RefinerLayers { get; init; }

    /// <summary>Caption feature width.</summary>
    public required int CaptionDim { get; init; }

    /// <summary>Patch input width (latent channels × patch area).</summary>
    public required int InChannels { get; init; }

    /// <summary>Timestep embedding width.</summary>
    public required int ModulationDim { get; init; }

    /// <summary>RoPE dimensions per position axis.</summary>
    public int[] AxesDim { get; init; } = [32, 48, 48];

    /// <summary>RoPE base.</summary>
    public float RopeTheta { get; init; } = 256f;

    /// <summary>RMS norm epsilon.</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>Latent channels.</summary>
    public int LatentChannels { get; init; } = 16;

    /// <summary>Flow scheduler: static shift.</summary>
    public float Shift { get; init; } = 3f;

    /// <summary>
    /// Whether the model is the few-step distilled Z-Image-Turbo (9 steps, no guidance; the default) or the undistilled Z-Image (50 steps,
    /// classifier-free guidance with a negative prompt (scale 5 in the usual convention, diffusers' 4), shift 6).
    /// </summary>
    public bool Distilled { get; init; } = true;

    /// <summary>The original scheduler configuration, or null.</summary>
    public string? SchedulerJson { get; init; }

    /// <summary>Infers the configuration from canonical <c>transformer.*</c> weights.</summary>
    /// <param name="weights">The canonical weights.</param>
    /// <exception cref="InvalidDataException">The weights are not a Z-Image transformer.</exception>
    public static ZImageConfig FromWeights(IReadOnlyDictionary<string, WeightInfo> weights)
    {
        if (!weights.TryGetValue("transformer.x_embedder.weight", out var embedder) || !weights.TryGetValue("transformer.layers.0.attention.norm_q.weight", out var norm))
        {
            throw new InvalidDataException("The checkpoint has no Z-Image transformer.");
        }

        int hidden = (int)embedder.Shape[0].Fixed;
        int headDim = (int)norm.Shape[0].Fixed;
        return new ZImageConfig
        {
            HiddenSize = hidden,
            HeadCount = hidden / headDim,
            Layers = ZImageNames.CountBlocks(weights, "layers"),
            RefinerLayers = ZImageNames.CountBlocks(weights, "noise_refiner"),
            CaptionDim = (int)weights["transformer.cap_embedder.linear.weight"].Shape[1].Fixed,
            InChannels = (int)embedder.Shape[1].Fixed,
            ModulationDim = (int)weights["transformer.layers.0.mod.weight"].Shape[1].Fixed,
            // Z-Image rotates 32 + 48 + 48 of its 128 head dimensions; files without a configuration keep those proportions.
            AxesDim = [headDim / 4, headDim * 3 / 8, headDim * 3 / 8],
        };
    }

    /// <summary>This configuration with the transformer and scheduler settings of a diffusers folder.</summary>
    /// <param name="schedulerJson">The <c>scheduler_config.json</c> text, or null.</param>
    /// <param name="transformerConfig">The <c>transformer/config.json</c>, or null.</param>
    public ZImageConfig WithDiffusers(string? schedulerJson, JsonElement? transformerConfig)
    {
        var result = this;
        if (transformerConfig is { } config)
        {
            result = result with
            {
                AxesDim = config.TryGetProperty("axes_dims", out _) ? config.Ints("axes_dims") : AxesDim,
                RopeTheta = config.Float("rope_theta", RopeTheta),
                NormEps = config.Float("norm_eps", NormEps),
            };
        }

        if (schedulerJson is null)
        {
            return result;
        }

        using var document = JsonDocument.Parse(schedulerJson);
        float shift = document.RootElement.Float("shift", 3f);

        // The Turbo's scheduler shifts by 3, the base model's by 6.
        return result with { Shift = shift, SchedulerJson = schedulerJson, Distilled = shift < 5f };
    }

    /// <summary>
    /// This configuration for a transformer file recognized by its name: files named like the base model (<c>z-image</c>, <c>z_image</c>
    /// without "turbo") are the undistilled Z-Image, everything else the Turbo (or a fine-tune of it).
    /// </summary>
    /// <param name="fileName">The file name.</param>
    public ZImageConfig WithFileName(string fileName)
        => IsBaseName(fileName) ? this with { Distilled = false, Shift = 6f } : this;

    /// <summary>Whether a model name is the undistilled Z-Image's (it starts with z-image / z_image and does not say turbo).</summary>
    /// <param name="name">A file or folder name.</param>
    public static bool IsBaseName(string name)
        => !name.Contains("turbo", StringComparison.OrdinalIgnoreCase)
            && (name.StartsWith("z-image", StringComparison.OrdinalIgnoreCase) || name.StartsWith("z_image", StringComparison.OrdinalIgnoreCase));

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
        metadata[t + "layers"] = Layers;
        metadata[t + "refiner_layers"] = RefinerLayers;
        metadata[t + "caption_dim"] = CaptionDim;
        metadata[t + "modulation_dim"] = ModulationDim;
        metadata[t + "axes_dim"] = AxesDim;
        metadata[t + "rope_theta"] = RopeTheta;
        metadata[t + "time_scale"] = 1000f;
        metadata[t + "norm_eps"] = NormEps;
        metadata[t + "f32_activations"] = !Distilled;
        metadata[t + "sample_size"] = 1024;
        string s = Prefix + "scheduler.";
        metadata[s + "train_steps"] = 1000;
        metadata[s + "shift"] = Shift;
        metadata[s + "dynamic_shift"] = false;
        metadata[s + "sampler"] = "euler";
        if (SchedulerJson is not null)
        {
            metadata[s + "source_config"] = SchedulerJson;
        }

        metadata[Prefix + "text_encoder.prompt_length"] = PromptLength;
        metadata[Prefix + "text_encoder.prompt_template"] = PromptTemplate;
        metadata[Prefix + "vae.scaling_factor"] = FluxConfig.VaeScalingFactor;
        metadata[Prefix + "vae.shift_factor"] = FluxConfig.VaeShiftFactor;
        metadata[Prefix + "vae.latent_rgb_factors"] = FluxConfig.RgbFactors;
        metadata[Prefix + "default_steps"] = Distilled ? 9 : 50;
        metadata[Prefix + "default_guidance"] = Distilled ? 0.0 : BaseGuidance;
        metadata[Prefix + "classifier_free_guidance"] = !Distilled;
    }
}
