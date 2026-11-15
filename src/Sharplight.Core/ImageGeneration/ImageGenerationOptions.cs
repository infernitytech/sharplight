namespace Sharplight.ImageGeneration;

/// <summary>Per-request settings of <see cref="IImageGenerator"/>; <see langword="null"/> values take the model's defaults.</summary>
public sealed record ImageGenerationOptions
{
    /// <summary>Output width in pixels, or <see langword="null"/> for the model's native size.</summary>
    public int? Width { get; init; }

    /// <summary>Output height in pixels, or <see langword="null"/> for the model's native size.</summary>
    public int? Height { get; init; }

    /// <summary>Number of denoising steps, or <see langword="null"/> for the model's default.</summary>
    public int? Steps { get; init; }

    /// <summary>Classifier-free guidance scale, or <see langword="null"/> for the model's default (distilled models ignore it).</summary>
    public float? GuidanceScale { get; init; }

    /// <summary>Seed of the initial noise; <see langword="null"/> picks one (reported in <see cref="ImageGenerationResult.Seed"/>).</summary>
    public long? Seed { get; init; }

    /// <summary>Number of images to generate.</summary>
    public int Count { get; init; } = 1;

    /// <summary>Scheduler (sampler) name understood by the implementation, or <see langword="null"/> for the model's configured one.</summary>
    public string? Scheduler { get; init; }

    /// <summary>
    /// Explicit initial noise instead of the seeded noise: unit Gaussian values of all images, image after image, each in the model's
    /// latent layout (for the latent-diffusion pipeline [channels, height / 8, width / 8]); used to reproduce another implementation's
    /// run exactly. <see langword="null"/> draws the noise from <see cref="Seed"/>.
    /// </summary>
    public float[]? InitialNoise { get; init; }

    /// <summary>
    /// Every how many steps the streaming call attaches a low-resolution preview of the current estimate to its update
    /// (<see cref="ImageGenerationUpdate.Preview"/>); 0 for none. Previews are cheap (no VAE decode) and need a model that
    /// describes its latent colours.
    /// </summary>
    public int PreviewInterval { get; init; }

    /// <summary>The defaults.</summary>
    public static ImageGenerationOptions Default { get; } = new();
}
