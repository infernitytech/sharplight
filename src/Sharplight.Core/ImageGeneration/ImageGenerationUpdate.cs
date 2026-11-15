namespace Sharplight.ImageGeneration;

/// <summary>Progress of <see cref="IImageGenerator.GenerateStreamingAsync"/>.</summary>
/// <param name="Step">Denoising steps completed.</param>
/// <param name="TotalSteps">Denoising steps of the request.</param>
/// <param name="Result">The finished images on the last update; <see langword="null"/> before.</param>
/// <param name="Preview">
/// A low-resolution preview of the image being generated (at latent resolution), when <see cref="ImageGenerationOptions.PreviewInterval"/>
/// asks for one at this step; otherwise <see langword="null"/>.
/// </param>
public sealed record ImageGenerationUpdate(int Step, int TotalSteps, ImageGenerationResult? Result = null, Vision.ImageInput? Preview = null);
