namespace Sharplight.VideoGeneration;

/// <summary>The resolved settings of one <see cref="VideoDiffusionGenerator"/> request.</summary>
internal sealed record VideoRun
{
    /// <summary>Output width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Output height in pixels.</summary>
    public required int Height { get; init; }

    /// <summary>Output frames (1 + k · temporal compression).</summary>
    public required int Frames { get; init; }

    /// <summary>Playback rate.</summary>
    public required double FramesPerSecond { get; init; }

    /// <summary>Latent channels.</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Latent frames.</summary>
    public required int LatentFrames { get; init; }

    /// <summary>Latent height.</summary>
    public required int LatentHeight { get; init; }

    /// <summary>Latent width.</summary>
    public required int LatentWidth { get; init; }

    /// <summary>Denoising steps.</summary>
    public required int Steps { get; init; }

    /// <summary>Classifier-free guidance scale.</summary>
    public required float Guidance { get; init; }

    /// <summary>Whether guidance runs (scale above 1): two denoiser runs per step.</summary>
    public required bool Guided { get; init; }

    /// <summary>Seed of the initial noise.</summary>
    public required long Seed { get; init; }

    /// <summary>Sampler name.</summary>
    public required string Sampler { get; init; }

    /// <summary>Preview every that many steps (0: none).</summary>
    public required int PreviewInterval { get; init; }

    /// <summary>Explicit initial noise, or <see langword="null"/>.</summary>
    public required float[]? InitialNoise { get; init; }

    /// <summary>Graph dimensions for a size other than the model's default, or <see langword="null"/>.</summary>
    public required IReadOnlyDictionary<string, long>? Dimensions { get; init; }

    /// <summary>Values of one latent plane (one channel of one latent frame).</summary>
    public int PlaneSize => LatentHeight * LatentWidth;

    /// <summary>Values of the latent video [C, T, h, w].</summary>
    public int LatentSize => LatentChannels * LatentFrames * LatentHeight * LatentWidth;
}
