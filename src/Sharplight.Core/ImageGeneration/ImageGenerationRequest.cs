using Sharplight.Vision;

namespace Sharplight.ImageGeneration;

/// <summary>What to generate (<see cref="IImageGenerator"/>).</summary>
public sealed record ImageGenerationRequest
{
    /// <summary>The text prompt.</summary>
    public required string Prompt { get; init; }

    /// <summary>What the image should not show (classifier-free guidance's negative conditioning), or <see langword="null"/>.</summary>
    public string? NegativePrompt { get; init; }

    /// <summary>An input image for image-to-image generation, or <see langword="null"/> for text-to-image.</summary>
    public ImageInput? Image { get; init; }

    /// <summary>For image-to-image: how much of <see cref="Image"/> is replaced, in [0, 1] (1 = ignore the input entirely).</summary>
    public float Strength { get; init; } = 0.75f;

    /// <summary>
    /// For inpainting: which parts of <see cref="Image"/> to regenerate (bright pixels; the mask is resized to the output and
    /// thresholded at half intensity per latent pixel), or <see langword="null"/> to regenerate the whole image. Requires
    /// <see cref="Image"/>.
    /// </summary>
    public ImageInput? Mask { get; init; }
}
