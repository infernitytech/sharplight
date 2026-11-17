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

    /// <summary>
    /// Model: the video is conditioned on a last frame as well (<see cref="VideoGenerationRequest.LastImage"/>; Wan 2.1 FLF2V), which
    /// image-to-video by channel concatenation then requires.
    /// </summary>
    public const string LastFrameConditioning = "lastFrameConditioning";

    /// <summary>
    /// Model (two experts): the fraction of the training timesteps from which down the second denoiser runs (Wan 2.2 A14B: 0.875; the
    /// first runs for timesteps ≥ ratio · train steps).
    /// </summary>
    public const string BoundaryRatio = "boundaryRatio";

    /// <summary>
    /// Model: the guidance scale is an input of the denoiser (<c>ImageGeneration.DiffusionGraphs.Guidance</c>, the scale · 1000; guidance-distilled
    /// HunyuanVideo) instead of classifier-free guidance; one denoiser evaluation per step.
    /// </summary>
    public const string EmbeddedGuidance = "embeddedGuidance";

    /// <summary>Text encoder: the prompt template, with <c>{}</c> where the prompt goes (HunyuanVideo's system instruction).</summary>
    public const string PromptTemplate = "promptTemplate";

    /// <summary>
    /// Text encoder: leading tokens of the templated prompt whose states are dropped (the template's instruction); the encoder graph takes
    /// these plus the context length and the denoiser the context length.
    /// </summary>
    public const string PromptPrefixTokens = "promptPrefixTokens";

    /// <summary>
    /// VAE (whole-video decoder): latent frames per decoded tile beyond the first; longer videos are decoded in overlapping tiles of
    /// one more latent frame, <see cref="DecodeTileStride"/> latent frames apart, blended over the overlap (diffusers' temporal tiling).
    /// </summary>
    public const string DecodeTileFrames = "decodeTileFrames";

    /// <summary>VAE: latent frames between the starts of consecutive decode tiles (<see cref="DecodeTileFrames"/>).</summary>
    public const string DecodeTileStride = "decodeTileStride";

    /// <summary>
    /// VAE (whole-video decoder): the side of square spatial decode tiles in latent pixels; larger videos are decoded in overlapping tiles
    /// <see cref="DecodeSpatialTileStride"/> latent pixels apart, blended over the overlap (diffusers' <c>enable_tiling</c>).
    /// </summary>
    public const string DecodeSpatialTile = "decodeSpatialTile";

    /// <summary>VAE: latent pixels between the starts of consecutive spatial decode tiles (<see cref="DecodeSpatialTile"/>).</summary>
    public const string DecodeSpatialTileStride = "decodeSpatialTileStride";

    /// <summary>Model (two experts): the default guidance scale of the second denoiser.</summary>
    public const string GuidanceScale2 = "guidanceScale2";
}
