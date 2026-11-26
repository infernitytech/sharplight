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

    /// <summary>Reference images of an editing request (0: none).</summary>
    public int References { get; init; }

    /// <summary>
    /// The image guidance scale of an InstructPix2Pix-style run (<see cref="ImageGenerationOptions.ImageGuidanceScale"/>); 0 for other
    /// runs. A guided run of such a model evaluates three rows (instruction with image, empty prompt with image, empty prompt without).
    /// </summary>
    public float ImageGuidance { get; init; }

    /// <summary>The aesthetic score of the prompt's row (models with five size conditions: the SDXL refiner).</summary>
    public float AestheticScore { get; init; }

    /// <summary>The aesthetic score of the negative prompt's row.</summary>
    public float NegativeAestheticScore { get; init; }

    /// <summary>Whether the image is decoded (and an input image encoded) by the tiny autoencoder.</summary>
    public bool TinyAutoencoder { get; init; }

    /// <summary>The scale of the ControlNet's residuals.</summary>
    public float ControlScale { get; init; } = 1f;

    /// <summary>The fraction of the steps before which the ControlNet does not apply.</summary>
    public float ControlStart { get; init; }

    /// <summary>The fraction of the steps after which the ControlNet does not apply.</summary>
    public float ControlEnd { get; init; } = 1f;

    /// <summary>Denoiser rows per evaluation: 1 unguided, 2 with classifier-free guidance, 3 with image guidance as well.</summary>
    public int Rows => !Guided ? 1 : ImageGuidance > 0 ? 3 : 2;

    /// <summary>Values of one latent image.</summary>
    public int LatentSize => LatentChannels * LatentHeight * LatentWidth;
}
