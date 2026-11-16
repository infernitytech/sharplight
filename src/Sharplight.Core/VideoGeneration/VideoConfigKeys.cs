namespace Sharplight.VideoGeneration;

/// <summary>
/// Configuration keys of video-generation model descriptions (in addition to the shared diffusion keys of
/// <see cref="ImageGeneration.DiffusionConfigKeys"/>: width, height, steps, guidance scale, latent channels, downscale, latent RGB factors,
/// context length, patch size and the scheduler keys).
/// </summary>
public static class VideoConfigKeys
{
    /// <summary>Model: default frame count.</summary>
    public const string Frames = "frames";

    /// <summary>Model: native playback rate in frames per second.</summary>
    public const string FramesPerSecond = "framesPerSecond";

    /// <summary>Model: the negative prompt used when a request gives none.</summary>
    public const string NegativePrompt = "negativePrompt";

    /// <summary>VAE: frames per latent frame after the first (temporal compression).</summary>
    public const string TemporalDownscale = "temporalDownscale";

    /// <summary>Transformer: the timestep is an input per token (the tokens of a conditioning first frame take 0).</summary>
    public const string TokenTimesteps = "tokenTimesteps";

    /// <summary>
    /// Model: image-to-video by replacing the first latent frame with the encoded start image (Wan 2.2 TI2V); needs the VAE encoder
    /// and <see cref="TokenTimesteps"/>.
    /// </summary>
    public const string FirstFrameConditioning = "firstFrameConditioning";
}
