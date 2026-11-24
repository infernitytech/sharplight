namespace Sharplight.VideoEnhancement;

/// <summary>Progress of <see cref="IVideoEnhancer.EnhanceStreamingAsync"/>.</summary>
/// <param name="Phase">What is running.</param>
/// <param name="Batch">The batch of frames the update belongs to (0-based).</param>
/// <param name="TotalBatches">Batches in the request.</param>
/// <param name="Frames">During <see cref="VideoEnhancementPhase.Decoding"/>: the finished frames of this batch, in order; otherwise <see langword="null"/>.</param>
/// <param name="Result">The whole result on the last update; <see langword="null"/> before.</param>
public sealed record VideoEnhancementUpdate(VideoEnhancementPhase Phase, int Batch, int TotalBatches, IReadOnlyList<Vision.ImageInput>? Frames = null,
    VideoEnhancementResult? Result = null);
