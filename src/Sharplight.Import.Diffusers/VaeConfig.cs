using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>Hyper-parameters of a KL autoencoder (the latent-diffusion VAE), written as the canonical <c>latent_diffusion.vae.*</c> keys.</summary>
internal sealed record VaeConfig
{
    /// <summary>Image channels (3).</summary>
    public int ImageChannels { get; init; } = 3;

    /// <summary>Latent channels.</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Channels per resolution level, finest first.</summary>
    public required int[] BlockChannels { get; init; }

    /// <summary>Residual blocks per encoder level (the decoder has one more per level).</summary>
    public required int LayersPerBlock { get; init; }

    /// <summary>GroupNorm groups.</summary>
    public int NormGroups { get; init; } = 32;

    /// <summary>Model latents are <c>(z - shift) * scale</c> of the encoder's mean.</summary>
    public required float ScalingFactor { get; init; }

    /// <summary>See <see cref="ScalingFactor"/>.</summary>
    public float ShiftFactor { get; init; }

    /// <summary>Whether the 1×1 convolutions between the encoder / decoder and the latent space exist.</summary>
    public bool QuantConvolutions { get; init; } = true;

    /// <summary>
    /// A linear map from model latents to approximate RGB in [-1, 1] (row-major [latent channels, 3]) for cheap previews, or empty.
    /// </summary>
    public float[] LatentRgbFactors { get; init; } = [];

    /// <summary>Writes the keys under <c>{prefix}vae.</c>.</summary>
    /// <param name="metadata">The canonical metadata.</param>
    /// <param name="prefix">The architecture prefix (<c>latent_diffusion.</c>).</param>
    public void WriteTo(IDictionary<string, object> metadata, string prefix)
    {
        string p = prefix + "vae.";
        metadata[p + "image_channels"] = ImageChannels;
        metadata[p + "latent_channels"] = LatentChannels;
        metadata[p + "block_channels"] = BlockChannels;
        metadata[p + "layers_per_block"] = LayersPerBlock;
        metadata[p + "norm_groups"] = NormGroups;
        metadata[p + "scaling_factor"] = ScalingFactor;
        metadata[p + "shift_factor"] = ShiftFactor;
        metadata[p + "quant_convolutions"] = QuantConvolutions;
        if (LatentRgbFactors.Length > 0)
        {
            metadata[p + "latent_rgb_factors"] = LatentRgbFactors;
        }
    }

    /// <summary>Reads a diffusers <c>vae/config.json</c>.</summary>
    /// <param name="config">The parsed configuration.</param>
    /// <param name="latentRgbFactors">Preview factors of the model family, or empty.</param>
    public static VaeConfig FromDiffusers(JsonElement config, float[] latentRgbFactors) => new()
    {
        ImageChannels = config.Int("in_channels", 3),
        LatentChannels = config.Int("latent_channels", 4),
        BlockChannels = config.Ints("block_out_channels"),
        LayersPerBlock = config.Int("layers_per_block", 2),
        NormGroups = config.Int("norm_num_groups", 32),
        ScalingFactor = config.Float("scaling_factor", 0.18215f),
        ShiftFactor = config.Float("shift_factor", 0),
        QuantConvolutions = config.Bool("use_quant_conv", true) && config.Bool("use_post_quant_conv", true),
        LatentRgbFactors = latentRgbFactors,
    };

    /// <summary>Infers the configuration from the canonical weights (<c>vae.*</c>).</summary>
    /// <param name="weights">The canonical weights.</param>
    /// <param name="scalingFactor">The latent scaling factor of the model family.</param>
    /// <param name="latentRgbFactors">Preview factors of the model family, or empty.</param>
    public static VaeConfig FromWeights(IReadOnlyDictionary<string, WeightInfo> weights, float scalingFactor, float[] latentRgbFactors)
    {
        var levels = new List<int>();
        for (int level = 0; weights.TryGetValue($"vae.encoder.down.{level}.res.0.conv1.weight", out var conv); level++)
        {
            levels.Add((int)conv.Shape[0].Fixed);
        }

        int layers = 0;
        while (weights.ContainsKey($"vae.encoder.down.0.res.{layers}.conv1.weight"))
        {
            layers++;
        }

        return new VaeConfig
        {
            ImageChannels = (int)weights["vae.encoder.input_conv.weight"].Shape[1].Fixed,
            LatentChannels = (int)weights["vae.decoder.input_conv.weight"].Shape[1].Fixed,
            BlockChannels = [.. levels],
            LayersPerBlock = layers,
            ScalingFactor = scalingFactor,
            QuantConvolutions = weights.ContainsKey("vae.post_quant_conv.weight"),
            LatentRgbFactors = latentRgbFactors,
        };
    }
}
