using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>Hyper-parameters of a Wan-style causal 3-D video VAE, read from <c>wan.vae.*</c>.</summary>
internal sealed record WanVaeHyperParameters
{
    /// <summary>Latent channels.</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Channel multipliers per level.</summary>
    public required int[] ChannelMultipliers { get; init; }

    /// <summary>Residual blocks per encoder level (the decoder has one more).</summary>
    public required int ResidualBlocks { get; init; }

    /// <summary>Whether each encoder level (but the last) halves the frame count; the decoder mirrors it.</summary>
    public required bool[] TemporalDownsample { get; init; }

    /// <summary>Residual up / down blocks with pixel-shuffle shortcuts (the Wan 2.2 VAE).</summary>
    public required bool Residual { get; init; }

    /// <summary>Pixel patching of the input and output (2 for the Wan 2.2 VAE: 2×2 pixels per channel group), or 1.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Per-channel latent mean (the pipeline's normalization).</summary>
    public required float[] LatentsMean { get; init; }

    /// <summary>Per-channel latent standard deviation.</summary>
    public required float[] LatentsStd { get; init; }

    /// <summary>Pixels per latent pixel.</summary>
    public required int SpatialDownscale { get; init; }

    /// <summary>Frames per latent frame after the first.</summary>
    public required int TemporalDownscale { get; init; }

    /// <summary>Reads the keys.</summary>
    /// <param name="source">The canonical source.</param>
    /// <param name="prefix">The architecture prefix (<c>wan.</c>).</param>
    public static WanVaeHyperParameters Read(ModelSource source, string prefix)
    {
        string p = prefix + "vae.";
        return new WanVaeHyperParameters
        {
            LatentChannels = source.Get<int>(p + "latent_channels"),
            ChannelMultipliers = source.Get<int[]>(p + "channel_multipliers", [1, 2, 4, 4]),
            ResidualBlocks = source.Get(p + "residual_blocks", 2),
            TemporalDownsample = source.Get<bool[]>(p + "temporal_downsample", [false, true, true]),
            Residual = source.Get(p + "residual", false),
            PatchSize = source.Get(p + "patch_size", 1),
            LatentsMean = source.Get<float[]>(p + "latents_mean"),
            LatentsStd = source.Get<float[]>(p + "latents_std"),
            SpatialDownscale = source.Get(p + "spatial_downscale", 8),
            TemporalDownscale = source.Get(p + "temporal_downscale", 4),
        };
    }
}
