using Sharplight.VideoGeneration;

namespace Sharplight.Architectures.VideoEnhancement;

/// <summary>
/// The graph contract between <see cref="SeedVrAdapter"/> and <see cref="SeedVrEnhancer"/>: graph, value and dimension names. Latent videos
/// are [C, T, h, w] (channels, latent frames, latent rows, latent columns); pixel clips are frames-first [F, 3, H, W] in [−1, 1].
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item><see cref="Restore"/> (component <c>transformer</c>): inputs <see cref="VideoDiffusionGraphs.Latent"/> f32 [C, T, h, w] (the noisy
/// latent), <see cref="Condition"/> f32 [C, T, h, w] (the encoded input clip) and <see cref="VideoDiffusionGraphs.Timestep"/> f32 [1]; output
/// <see cref="VideoDiffusionGraphs.Prediction"/> f32 [C, T, h, w], the velocity (the restored latent is latent − timestep / 1000 · prediction).
/// The text conditioning is the model's precomputed positive embedding, a weight.</item>
/// <item><see cref="VaeEncode"/> (component <c>vae</c>): input <see cref="VideoDiffusionGraphs.Frames"/> f32 [F, 3, H, W] in [−1, 1]
/// (F = 1 + 4k); outputs <see cref="Mean"/> and <see cref="Deviation"/> f32 [C, T, h, w], the latent distribution scaled by the latent
/// scaling factor.</item>
/// <item><see cref="VaeDecode"/> (component <c>vae</c>): input <see cref="VideoDiffusionGraphs.Latent"/> f32 [C, T, h, w] (scaled);
/// output <see cref="VideoDiffusionGraphs.Frames"/> f32 [F, 3, H, W] in about [−1, 1], not clamped.</item>
/// <item><see cref="ColorCorrect"/> (component <c>vae</c>): inputs <see cref="VideoDiffusionGraphs.Frames"/> (the decoded frames) and
/// <see cref="Reference"/> (the resized input) f32 [F, 3, H, W] in [−1, 1]; output <see cref="Output"/> f32 [F, 3, H, W] in [0, 1]: the
/// wavelet colour transfer, clamped.</item>
/// </list>
/// Sizes are fixed per graph; the model is rebuilt for other sizes with <see cref="VideoDiffusionGraphs.FramesDimension"/> (latent frames),
/// <see cref="VideoDiffusionGraphs.Height"/> and <see cref="VideoDiffusionGraphs.Width"/> (pixels).
/// </remarks>
internal static class SeedVrGraphs
{
    /// <summary>The restoration DiT graph.</summary>
    public const string Restore = "restore";

    /// <summary>The VAE encoder graph of a clip.</summary>
    public const string VaeEncode = "vae_encode";

    /// <summary>The VAE decoder graph of a latent clip.</summary>
    public const string VaeDecode = "vae_decode";

    /// <summary>The wavelet colour-correction graph.</summary>
    public const string ColorCorrect = "color_correct";

    /// <summary>Input of <see cref="Restore"/>: the encoded input clip.</summary>
    public const string Condition = "condition";

    /// <summary>Output of <see cref="VaeEncode"/>: the scaled mean of the latent distribution.</summary>
    public const string Mean = "mean";

    /// <summary>Output of <see cref="VaeEncode"/>: the scaled standard deviation of the latent distribution.</summary>
    public const string Deviation = "deviation";

    /// <summary>Input of <see cref="ColorCorrect"/>: the frames whose colours the result takes.</summary>
    public const string Reference = "reference";

    /// <summary>Output of <see cref="ColorCorrect"/>.</summary>
    public const string Output = "output";
}
