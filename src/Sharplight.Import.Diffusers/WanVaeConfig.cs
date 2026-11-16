using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The configuration of a Wan-style causal 3-D VAE (Wan 2.1 / 2.2, Qwen-Image's <c>AutoencoderKLQwenImage</c>) in canonical keys
/// (<c>{prefix}vae.*</c>, read by the <c>WanVaeHyperParameters</c> of Sharplight.Core): from a diffusers <c>config.json</c>, or from
/// the shapes of the canonical <c>vae.*</c> weights for stand-alone files without one (16 latent channels: Wan 2.1's latent
/// normalization, which Qwen-Image shares).
/// </summary>
internal static class WanVaeConfig
{
    /// <summary>The per-channel latent mean of the Wan 2.1 VAE (and Qwen-Image's).</summary>
    public static readonly float[] Wan21LatentsMean =
        [-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f, 0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f];

    /// <summary>The per-channel latent standard deviation of the Wan 2.1 VAE (and Qwen-Image's).</summary>
    public static readonly float[] Wan21LatentsStd =
        [2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f, 3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.916f];

    /// <summary>Writes the keys of a diffusers VAE configuration.</summary>
    /// <param name="vae">The VAE's <c>config.json</c>.</param>
    /// <param name="metadata">The canonical metadata.</param>
    /// <param name="prefix">The architecture prefix (<c>wan.</c>, <c>qwen_image.</c>).</param>
    public static void WriteTo(JsonElement vae, IDictionary<string, object> metadata, string prefix)
    {
        string v = prefix + "vae.";
        metadata[v + "latent_channels"] = vae.Int("z_dim", 16);
        metadata[v + "channel_multipliers"] = vae.TryGetProperty("dim_mult", out _) ? vae.Ints("dim_mult") : [1, 2, 4, 4];
        metadata[v + "residual_blocks"] = vae.Int("num_res_blocks", 2);
        metadata[v + "temporal_downsample"] = vae.TryGetProperty("temperal_downsample", out var temporal)
            ? temporal.EnumerateArray().Select(e => e.GetBoolean()).ToArray()
            : new[] { false, true, true };
        metadata[v + "residual"] = vae.Bool("is_residual", false);
        metadata[v + "patch_size"] = vae.Int("patch_size", 1);
        metadata[v + "latents_mean"] = Floats(vae, "latents_mean");
        metadata[v + "latents_std"] = Floats(vae, "latents_std");
        metadata[v + "spatial_downscale"] = vae.Int("scale_factor_spatial", 8);
        metadata[v + "temporal_downscale"] = vae.Int("scale_factor_temporal", 4);
    }

    /// <summary>
    /// Writes the keys inferred from the canonical weights of a non-residual (Wan 2.1-layout) VAE: latent channels, channel multipliers,
    /// residual blocks per level and the levels that halve the frames (their resampler has a <c>time_conv</c>).
    /// </summary>
    /// <param name="weights">The canonical weights (<c>vae.*</c>, diffusers names).</param>
    /// <param name="metadata">The canonical metadata.</param>
    /// <param name="prefix">The architecture prefix.</param>
    /// <exception cref="InvalidDataException">The weights are not a Wan 2.1-layout VAE, or it has other than 16 latent channels.</exception>
    public static void WriteFromWeights(IReadOnlyDictionary<string, WeightInfo> weights, IDictionary<string, object> metadata, string prefix)
    {
        const string p = WanNames.VaePrefix;
        if (!weights.TryGetValue(p + "post_quant_conv.weight", out var post) || !weights.TryGetValue(p + "encoder.conv_in.weight", out var convIn))
        {
            throw new InvalidDataException("The VAE has no quant convolutions or encoder input convolution.");
        }

        int z = (int)post.Shape[0].Fixed;
        if (z != 16)
        {
            throw new InvalidDataException($"A VAE file with {z} latent channels needs its config.json (latent normalization) beside it.");
        }

        long baseDim = convIn.Shape[0].Fixed;
        var multipliers = new List<int>();
        var temporal = new List<bool>();
        int index = 0;
        int residualBlocks = 0;
        while (weights.ContainsKey($"{p}encoder.down_blocks.{index}.conv1.weight") || weights.ContainsKey($"{p}encoder.down_blocks.{index}.resample.1.weight"))
        {
            int blocks = 0;
            long channels = 0;
            while (weights.TryGetValue($"{p}encoder.down_blocks.{index}.conv1.weight", out var conv))
            {
                channels = conv.Shape[0].Fixed;
                blocks++;
                index++;
            }

            residualBlocks = Math.Max(residualBlocks, blocks);
            multipliers.Add((int)(channels / baseDim));
            if (weights.ContainsKey($"{p}encoder.down_blocks.{index}.resample.1.weight"))
            {
                temporal.Add(weights.ContainsKey($"{p}encoder.down_blocks.{index}.time_conv.weight"));
                index++;
            }
        }

        string v = prefix + "vae.";
        metadata[v + "latent_channels"] = z;
        metadata[v + "channel_multipliers"] = multipliers.ToArray();
        metadata[v + "residual_blocks"] = residualBlocks;
        metadata[v + "temporal_downsample"] = temporal.ToArray();
        metadata[v + "residual"] = false;
        metadata[v + "patch_size"] = 1;
        metadata[v + "latents_mean"] = Wan21LatentsMean;
        metadata[v + "latents_std"] = Wan21LatentsStd;
        metadata[v + "spatial_downscale"] = 1 << temporal.Count;
        metadata[v + "temporal_downscale"] = 1 << temporal.Count(t => t);
    }

    private static float[] Floats(JsonElement config, string key)
        => config.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Array
            ? [.. value.EnumerateArray().Select(e => e.GetSingle())]
            : throw new InvalidDataException($"The VAE configuration has no '{key}' list.");
}
