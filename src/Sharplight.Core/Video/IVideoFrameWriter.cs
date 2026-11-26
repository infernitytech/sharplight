using Sharplight.Vision;

namespace Sharplight.Video;

/// <summary>
/// An encode that takes its frames one at a time (<see cref="IVideoEncoder.OpenWriter"/>, <see cref="VideoEncoderCatalog.OpenWriter"/>): each
/// frame is encoded and written as it arrives, so the memory it needs does not grow with the length of the video. Not thread safe: one
/// call at a time. <see cref="CompleteAsync"/> finishes the file; disposing without completing aborts the encode (the output is not a
/// finished video, and a writer of <see cref="VideoEncoderCatalog.OpenWriter"/> deletes its file).
/// </summary>
public interface IVideoFrameWriter : IAsyncDisposable
{
    /// <summary>The frames written so far.</summary>
    public long FramesWritten { get; }

    /// <summary>Encodes one frame (the next in display order); the CPU work runs on the thread pool.</summary>
    /// <param name="frame">8-bit RGB with the size of <see cref="VideoStreamSettings.Width"/> × <see cref="VideoStreamSettings.Height"/>; not kept after the call.</param>
    /// <param name="cancellationToken">Cancels the wait; a cancelled write leaves the writer usable for <see cref="CompleteAsync"/>.</param>
    /// <exception cref="ArgumentException">The frame has another size.</exception>
    /// <exception cref="InvalidOperationException">The writer is completed.</exception>
    public ValueTask WriteAsync(ImageInput frame, CancellationToken cancellationToken = default);

    /// <summary>
    /// Finishes the file: flushes the encoder, writes the audio up to the end of the video and the container's trailer. The result is a
    /// playable video of the <see cref="FramesWritten"/> frames (also after an interrupted run). Further calls do nothing.
    /// </summary>
    /// <param name="cancellationToken">Cancels before the finishing starts.</param>
    public Task CompleteAsync(CancellationToken cancellationToken = default);
}
