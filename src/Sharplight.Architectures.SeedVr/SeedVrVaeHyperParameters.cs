using Sharplight.Models;

namespace Sharplight.Architectures.SeedVr;

/// <summary>Hyper-parameters of the SeedVR2 causal video VAE, read from <c>seedvr.vae.*</c>.</summary>
internal sealed record SeedVrVaeHyperParameters
{
    /// <summary>Channels per resolution level, from the full resolution down.</summary>
    public required int[] BlockChannels { get; init; }

    /// <summary>Residual blocks per encoder level (the decoder has one more).</summary>
    public required int LayersPerBlock { get; init; }

    /// <summary>Latent channels.</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Group-norm groups.</summary>
    public required int Groups { get; init; }

    /// <summary>Epsilon of the group norms.</summary>
    public required float Eps { get; init; }

    /// <summary>The latent scaling factor (latents are multiplied by it after encoding).</summary>
    public required float ScalingFactor { get; init; }

    /// <summary>Temporal compression (frames per latent frame after the first).</summary>
    public required int TemporalDownscale { get; init; }

    /// <summary>Spatial compression.</summary>
    public required int SpatialDownscale { get; init; }

    /// <summary>Reads the hyper-parameters.</summary>
    /// <param name="src">The canonical source.</param>
    /// <param name="prefix">The architecture prefix (<c>seedvr.</c>).</param>
    public static SeedVrVaeHyperParameters Read(ModelSource src, string prefix)
    {
        string p = prefix + "vae.";
        return new SeedVrVaeHyperParameters
        {
            BlockChannels = src.Get<int[]>(p + "block_channels"),
            LayersPerBlock = src.Get(p + "layers_per_block", 2),
            LatentChannels = src.Get(p + "latent_channels", 16),
            Groups = src.Get(p + "groups", 32),
            Eps = src.Get(p + "eps", 1e-6f),
            ScalingFactor = src.Get(p + "scaling_factor", 0.9152f),
            TemporalDownscale = src.Get(p + "temporal_downscale", 4),
            SpatialDownscale = src.Get(p + "spatial_downscale", 8),
        };
    }
}
