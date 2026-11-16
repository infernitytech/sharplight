using System.Text.Json;
using Sharplight.ImageGeneration;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The configuration of a Wan video model in canonical keys (<c>wan.*</c>): the transformer and VAE hyper-parameters, the scheduler
/// and the pipeline defaults (Wan 2.1: 832×480, 81 frames at 16 fps; Wan 2.2 TI2V: 1280×704, 121 frames at 24 fps; 50 steps,
/// guidance 5).
/// </summary>
internal sealed record WanConfig
{
    /// <summary>The canonical architecture id (the <c>WanAdapter</c>'s).</summary>
    public const string Architecture = "wan";

    /// <summary>Metadata key prefix.</summary>
    public const string Prefix = Architecture + ".";

    /// <summary>Canonical metadata of the transformer, the VAE, the scheduler and the defaults.</summary>
    public required IReadOnlyDictionary<string, object> Metadata { get; init; }

    /// <summary>Reads diffusers configurations.</summary>
    /// <param name="transformer">The transformer's <c>config.json</c>.</param>
    /// <param name="vae">The VAE's <c>config.json</c>.</param>
    /// <param name="scheduler">The scheduler's <c>scheduler_config.json</c> text, or null.</param>
    /// <param name="pipeline">The folder's <c>model_index.json</c>, or null.</param>
    /// <exception cref="NotSupportedException">An image-conditioned Wan 2.1 I2V transformer (CLIP image embedding), not supported yet.</exception>
    public static WanConfig FromDiffusers(JsonElement transformer, JsonElement vae, string? scheduler, JsonElement? pipeline)
    {
        if (transformer.TryGetProperty("image_dim", out var imageDim) && imageDim.ValueKind == JsonValueKind.Number)
        {
            throw new NotSupportedException("Not supported yet: Wan image-to-video transformers with a CLIP image embedding (Wan 2.1 I2V / FLF2V); "
                + "Wan 2.1 text-to-video and Wan 2.2 TI2V are supported.");
        }

        int heads = transformer.Int("num_attention_heads", 40);
        int headDim = transformer.Int("attention_head_dim", 128);
        bool expandTimesteps = pipeline is { } index && index.Bool("expand_timesteps", false);
        bool residual = vae.Bool("is_residual", false);
        var metadata = new Dictionary<string, object>(StringComparer.Ordinal);
        string t = Prefix + "transformer.";
        metadata[t + "in_channels"] = transformer.Int("in_channels", 16);
        metadata[t + "out_channels"] = transformer.Int("out_channels", transformer.Int("in_channels", 16));
        metadata[t + "patch_size"] = transformer.IntOrList("patch_size", 2, 3);
        metadata[t + "hidden_size"] = heads * headDim;
        metadata[t + "head_count"] = heads;
        metadata[t + "layer_count"] = transformer.Int("num_layers", 40);
        metadata[t + "ffn_dim"] = transformer.Int("ffn_dim", 13824);
        metadata[t + "text_dim"] = transformer.Int("text_dim", 4096);
        metadata[t + "freq_dim"] = transformer.Int("freq_dim", 256);
        metadata[t + "norm_eps"] = transformer.Float("eps", 1e-6f);
        metadata[t + "cross_attn_norm"] = transformer.Bool("cross_attn_norm", true);
        metadata[t + "token_timesteps"] = expandTimesteps;

        string v = Prefix + "vae.";
        int[] multipliers = vae.TryGetProperty("dim_mult", out _) ? vae.Ints("dim_mult") : [1, 2, 4, 4];
        metadata[v + "latent_channels"] = vae.Int("z_dim", 16);
        metadata[v + "channel_multipliers"] = multipliers;
        metadata[v + "residual_blocks"] = vae.Int("num_res_blocks", 2);
        metadata[v + "temporal_downsample"] = vae.TryGetProperty("temperal_downsample", out var temporal)
            ? temporal.EnumerateArray().Select(e => e.GetBoolean()).ToArray()
            : new[] { false, true, true };
        metadata[v + "residual"] = residual;
        metadata[v + "patch_size"] = vae.Int("patch_size", 1);
        metadata[v + "latents_mean"] = Floats(vae, "latents_mean");
        metadata[v + "latents_std"] = Floats(vae, "latents_std");
        metadata[v + "spatial_downscale"] = vae.Int("scale_factor_spatial", 8);
        metadata[v + "temporal_downscale"] = vae.Int("scale_factor_temporal", 4);

        string s = Prefix + "scheduler.";
        string sampler = Samplers.UniPc;
        float shift = residual ? 5f : 3f;
        int trainSteps = 1000;
        if (scheduler is not null)
        {
            using var document = JsonDocument.Parse(scheduler);
            var root = document.RootElement;
            sampler = root.String("_class_name", "") == "FlowMatchEulerDiscreteScheduler" ? Samplers.Euler : Samplers.UniPc;
            shift = root.Float(sampler == Samplers.Euler ? "shift" : "flow_shift", shift);
            trainSteps = root.Int("num_train_timesteps", trainSteps);
            metadata[s + "source_config"] = scheduler;
        }

        metadata[s + "sampler"] = sampler;
        metadata[s + "shift"] = shift;
        metadata[s + "train_steps"] = trainSteps;
        metadata[Prefix + "default_width"] = residual ? 1280 : 832;
        metadata[Prefix + "default_height"] = residual ? 704 : 480;
        metadata[Prefix + "default_frames"] = residual ? 121 : 81;
        metadata[Prefix + "frames_per_second"] = residual ? 24.0 : 16.0;
        metadata[Prefix + "default_steps"] = 50;
        metadata[Prefix + "default_guidance"] = 5.0;
        metadata[Prefix + "first_frame_conditioning"] = expandTimesteps;
        return new WanConfig { Metadata = metadata };
    }

    private static float[] Floats(JsonElement config, string key)
        => config.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Array
            ? [.. value.EnumerateArray().Select(e => e.GetSingle())]
            : throw new InvalidDataException($"The VAE configuration has no '{key}' list.");
}
