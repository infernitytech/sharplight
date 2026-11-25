namespace Sharplight.VideoEnhancement;

/// <summary>
/// Progress of <see cref="IVideoEnhancer.EnhanceStreamingAsync"/>: an update when a stage of a batch starts, one with the batch's frames when its
/// decoding has finished, and a last one with the result.
/// </summary>
/// <param name="Phase">What is running.</param>
/// <param name="Batch">The batch of frames the update belongs to (0-based).</param>
/// <param name="TotalBatches">Batches in the request.</param>
/// <param name="Result">The whole result on the last update; <see langword="null"/> before.</param>
/// <param name="Frames">The finished frames of the batch, in order, on the <see cref="VideoEnhancementPhase.Decoding"/> update that ends its decoding; otherwise <see langword="null"/>.</param>
public sealed record VideoEnhancementUpdate(VideoEnhancementPhase Phase, int Batch, int TotalBatches, VideoEnhancementResult? Result = null,
    IReadOnlyList<Vision.ImageInput>? Frames = null);
