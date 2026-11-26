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

    /// <summary>
    /// For editing models (the denoiser's <see cref="DiffusionConfigKeys.ReferenceImages"/>; FLUX.1 Kontext, FLUX.2): the images the prompt
    /// describes changes to (the instruction), resized to the output size; the output size defaults to the first one's. <see langword="null"/>
    /// or empty for text-to-image.
    /// </summary>
    public IReadOnlyList<ImageInput>? ReferenceImages { get; init; }

    /// <summary>
    /// For models with a ControlNet (<see cref="ImageGenerationDefaults.ControlNet"/>): the control image (a canny edge map, depth map, pose
    /// skeleton, ... as the ControlNet was trained on), resized to the output size; the output size defaults to its size. <see langword="null"/>
    /// to generate without control (the ControlNet is not run).
    /// </summary>
    public ImageInput? ControlImage { get; init; }
}
