using Sharplight.Vision;

namespace Sharplight.VideoGeneration;

/// <summary>Progress of <see cref="IVideoGenerator.GenerateStreamingAsync"/>.</summary>
/// <param name="Phase">What is running.</param>
/// <param name="Step">Units of <paramref name="Phase"/> completed (denoising steps, or decoded frames).</param>
/// <param name="TotalSteps">Units of <paramref name="Phase"/> in the request.</param>
/// <param name="Result">The finished video on the last update; <see langword="null"/> before.</param>
/// <param name="Preview">
/// Low-resolution preview frames (one per latent frame, at latent resolution) of the current estimate when
/// <see cref="VideoGenerationOptions.PreviewInterval"/> asks for them at this step; otherwise <see langword="null"/>.
/// </param>
/// <param name="Frames">During <see cref="VideoGenerationPhase.Decoding"/>: the frames decoded by this update, in order; otherwise <see langword="null"/>.</param>
public sealed record VideoGenerationUpdate(VideoGenerationPhase Phase, int Step, int TotalSteps, VideoGenerationResult? Result = null,
    IReadOnlyList<ImageInput>? Preview = null, IReadOnlyList<ImageInput>? Frames = null);
