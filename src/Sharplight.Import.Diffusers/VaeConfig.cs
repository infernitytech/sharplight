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
    /// <summary>Latents are <c>(z - shift) * scale</c> of the encoder's mean.</summary>
    public required float ScalingFactor { get; init; }
    /// <summary>See <see cref="ScalingFactor"/>.</summary>
    public float ShiftFactor { get; init; }

    /// <summary>Writes the keys under <c>{prefix}vae.</c>.</summary>
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
    }
}
