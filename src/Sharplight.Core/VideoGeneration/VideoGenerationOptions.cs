namespace Sharplight.VideoGeneration;

/// <summary>Per-request settings of <see cref="IVideoGenerator"/>; <see langword="null"/> values take the model's defaults (<see cref="VideoGenerationDefaults"/>).</summary>
public sealed record VideoGenerationOptions
{
    /// <summary>Output width in pixels (rounded down to the model's multiple), or <see langword="null"/> for the model's default.</summary>
    public int? Width { get; init; }

    /// <summary>Output height in pixels (rounded down to the model's multiple), or <see langword="null"/> for the model's default.</summary>
    public int? Height { get; init; }

    /// <summary>
    /// Number of frames, or <see langword="null"/> for the model's default. Causal video VAEs decode 1 + k·r frames (r the temporal
    /// compression, 4 for Wan); other counts are rounded down to the nearest such count.
    /// </summary>
    public int? Frames { get; init; }

    /// <summary>Playback rate of the result in frames per second, or <see langword="null"/> for the model's native rate (16 for Wan 2.1).</summary>
    public double? FramesPerSecond { get; init; }

    /// <summary>Number of denoising steps, or <see langword="null"/> for the model's default.</summary>
    public int? Steps { get; init; }

    /// <summary>Classifier-free guidance scale, or <see langword="null"/> for the model's default (1 or less: unguided, half the work).</summary>
    public float? GuidanceScale { get; init; }

    /// <summary>Seed of the initial noise; <see langword="null"/> picks one (reported in <see cref="VideoGenerationResult.Seed"/>).</summary>
    public long? Seed { get; init; }

    /// <summary>Scheduler (sampler) name, or <see langword="null"/> for the model's configured one (<see cref="ImageGeneration.Samplers"/> names).</summary>
    public string? Scheduler { get; init; }

    /// <summary>
    /// Explicit initial noise instead of the seeded noise: unit Gaussian values in the model's latent layout [channels, latent frames,
    /// height / s, width / s]; used to reproduce another implementation's run exactly. <see langword="null"/> draws it from <see cref="Seed"/>.
    /// </summary>
    public float[]? InitialNoise { get; init; }

    /// <summary>
    /// Every how many steps the streaming call attaches low-resolution preview frames of the current estimate to its update
    /// (<see cref="VideoGenerationUpdate.Preview"/>); 0 for none. Previews are cheap (no VAE decode) and need a model that describes its
    /// latent colours.
    /// </summary>
    public int PreviewInterval { get; init; }

    /// <summary>The defaults.</summary>
    public static VideoGenerationOptions Default { get; } = new();
}
