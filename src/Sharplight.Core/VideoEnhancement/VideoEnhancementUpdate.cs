using Sharplight.Vision;

namespace Sharplight.VideoEnhancement;

/// <summary>
/// Progress of <see cref="IVideoEnhancer.EnhanceStreamingAsync"/>: an update when a stage of a batch starts, one with the batch's finished
/// frames when its decoding has ended, and a last one (<see cref="VideoEnhancementPhase.Completed"/>) when the clip is done. The stream
/// never collects the clip: a consumer that needs every frame keeps them itself (<see cref="IVideoEnhancer.EnhanceAsync"/> does).
/// </summary>
/// <param name="Phase">What is running.</param>
/// <param name="Batch">The batch of frames the update belongs to (0-based).</param>
/// <param name="TotalBatches">
/// Batches in the request: exact for frames in memory, estimated from <see cref="VideoEnhancementRequest.FrameCount"/> for a frame source
/// (exact on the <see cref="VideoEnhancementPhase.Completed"/> update), 0 while unknown.
/// </param>
public sealed record VideoEnhancementUpdate(VideoEnhancementPhase Phase, int Batch, int TotalBatches)
{
    /// <summary>
    /// The finished frames of the batch, in order, on the <see cref="VideoEnhancementPhase.Decoding"/> update that ends its decoding; otherwise
    /// <see langword="null"/>. Owned by the consumer.
    /// </summary>
    public IReadOnlyList<ImageInput>? Frames { get; init; }

    /// <summary>The clip index of <see cref="Frames"/>[0] (0-based); the index of the batch's first frame on the other updates.</summary>
    public long FirstFrame { get; init; }

    /// <summary>The frames finished so far, these <see cref="Frames"/> included.</summary>
    public long FinishedFrames { get; init; }

    /// <summary>
    /// Frames in the clip: exact for frames in memory and on the <see cref="VideoEnhancementPhase.Completed"/> update, else
    /// <see cref="VideoEnhancementRequest.FrameCount"/> (<see langword="null"/> when the request did not say).
    /// </summary>
    public long? TotalFrames { get; init; }

    /// <summary>The seed of the noise the model started from (models without noise report 0).</summary>
    public long Seed { get; init; }

    /// <summary>The playback rate of the result (the request's, or 24 when it gave none).</summary>
    public double FramesPerSecond { get; init; }

    /// <summary>The size of the enhanced frames in pixels (both 0 before the request's first frame is known).</summary>
    public (int Width, int Height) Size { get; init; }
}
