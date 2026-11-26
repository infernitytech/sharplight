using Sharplight.Vision;

namespace Sharplight.Video.Ffmpeg;

/// <summary>The <see cref="IVideoFrameWriter"/> of <see cref="FfmpegVideoEncoder"/>: runs each step of an <see cref="FfmpegVideoWriter"/> on the thread pool, one at a time.</summary>
internal sealed class FfmpegVideoFrameWriter(FfmpegVideoWriter writer, int width, int height) : IVideoFrameWriter
{
    private readonly SemaphoreSlim _gate = new(1, 1);
    private bool _completed;
    private bool _disposed;

    /// <inheritdoc />
    public long FramesWritten => writer.FramesWritten;

    /// <inheritdoc />
    public async ValueTask WriteAsync(ImageInput frame, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if (frame.Width != width || frame.Height != height || frame.Rgb.Length != 3 * width * height)
        {
            throw new ArgumentException($"The frame is {frame.Width}x{frame.Height} ({frame.Rgb.Length} bytes), the video {width}x{height} RGB.", nameof(frame));
        }

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_completed)
            {
                throw new InvalidOperationException("The video is completed.");
            }

            await Task.Run(() => writer.Add(frame), CancellationToken.None).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <inheritdoc />
    public async Task CompleteAsync(CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_completed)
            {
                return;
            }

            await Task.Run(writer.Finish, CancellationToken.None).ConfigureAwait(false);
            _completed = true;
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        await _gate.WaitAsync().ConfigureAwait(false);
        try
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            writer.Dispose();
        }
        finally
        {
            _gate.Release();
        }
    }
}
