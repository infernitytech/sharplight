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

    /// <summary>The defaults.</summary>
    public static ImageGenerationOptions Default { get; } = new();
}
