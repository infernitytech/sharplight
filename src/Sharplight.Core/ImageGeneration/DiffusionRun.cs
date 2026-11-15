namespace Sharplight.ImageGeneration;

/// <summary>The resolved settings of one <see cref="LatentDiffusionGenerator"/> request.</summary>
internal sealed record DiffusionRun
{
    /// <summary>Output width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Output height in pixels.</summary>
    public required int Height { get; init; }

    /// <summary>Latent width.</summary>
    public required int LatentWidth { get; init; }

    /// <summary>Latent height.</summary>
    public required int LatentHeight { get; init; }

    /// <summary>Latent channels.</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Denoising steps.</summary>
    public required int Steps { get; init; }

    /// <summary>Classifier-free guidance scale.</summary>
    public required float Guidance { get; init; }

    /// <summary>Whether guidance runs (scale above 1): two denoiser rows per evaluation.</summary>
    public required bool Guided { get; init; }

    /// <summary>Seed of the first image.</summary>
    public required long Seed { get; init; }

    /// <summary>Images to generate.</summary>
    public required int Count { get; init; }

    /// <summary>Sampler name.</summary>
    public required string Sampler { get; init; }

    /// <summary>Preview every that many steps (0: none).</summary>
    public required int PreviewInterval { get; init; }

    /// <summary>Explicit initial noise, or <see langword="null"/>.</summary>
    public required float[]? InitialNoise { get; init; }

    /// <summary>Graph dimensions for a non-native size, or <see langword="null"/>.</summary>
    public required IReadOnlyDictionary<string, long>? Dimensions { get; init; }

    /// <summary>Values of one latent image.</summary>
    public int LatentSize => LatentChannels * LatentHeight * LatentWidth;
}
