using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a KL autoencoder (canonical keys <c>latent_diffusion.vae.*</c>).</summary>
internal sealed record AutoencoderHyperParameters
{
    /// <summary>Image channels.</summary>
    public required int ImageChannels { get; init; }

    /// <summary>Latent channels.</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Channels per level, finest first.</summary>
    public required int[] BlockChannels { get; init; }

    /// <summary>Residual blocks per encoder level (one more per decoder level).</summary>
    public required int LayersPerBlock { get; init; }

    /// <summary>GroupNorm groups.</summary>
    public required int NormGroups { get; init; }

    /// <summary>Model latents are <c>(z - shift) · scale</c>.</summary>
    public required float ScalingFactor { get; init; }

    /// <summary>See <see cref="ScalingFactor"/>.</summary>
    public required float ShiftFactor { get; init; }

    /// <summary>Whether the 1×1 quant / post-quant convolutions exist.</summary>
    public required bool QuantConvolutions { get; init; }

    /// <summary>Latent-to-RGB preview factors ([latent channels, 3]), or empty.</summary>
    public required float[] LatentRgbFactors { get; init; }

    /// <summary>Pixels per latent along each axis.</summary>
    public int Downscale => 1 << (BlockChannels.Length - 1);

    /// <summary>Reads the keys under <c>{prefix}vae.</c>.</summary>
    /// <param name="source">The canonical source.</param>
    /// <param name="prefix">Architecture prefix (<c>latent_diffusion.</c>).</param>
    public static AutoencoderHyperParameters Read(ModelSource source, string prefix)
    {
        string p = prefix + "vae.";
        return new AutoencoderHyperParameters
        {
            ImageChannels = source.Get(p + "image_channels", 3),
            LatentChannels = source.Get<int>(p + "latent_channels"),
            BlockChannels = source.Get<int[]>(p + "block_channels"),
            LayersPerBlock = source.Get<int>(p + "layers_per_block"),
            NormGroups = source.Get(p + "norm_groups", 32),
            ScalingFactor = source.Get<float>(p + "scaling_factor"),
            ShiftFactor = source.Get(p + "shift_factor", 0f),
            QuantConvolutions = source.Get(p + "quant_convolutions", true),
            LatentRgbFactors = source.Get<float[]>(p + "latent_rgb_factors", []),
        };
    }
}
