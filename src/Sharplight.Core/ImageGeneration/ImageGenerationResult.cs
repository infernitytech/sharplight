using Sharplight.Vision;

namespace Sharplight.ImageGeneration;

/// <summary>The result of <see cref="IImageGenerator.GenerateAsync"/>.</summary>
/// <param name="Images">The generated 8-bit RGB images (<see cref="Imaging.PngCodec"/> saves them).</param>
/// <param name="Seed">The seed of the first image's noise; image i used <c>Seed + i</c>.</param>
public sealed record ImageGenerationResult(IReadOnlyList<ImageInput> Images, long Seed);
