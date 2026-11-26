using Sharplight.Vision;

namespace Sharplight.Video;

/// <summary>The writer of <see cref="VideoEncoderCatalog.OpenWriter"/>: owns the file, deletes it when disposed without being completed.</summary>
internal sealed class FileVideoFrameWriter(IVideoFrameWriter inner, FileStream file, string path) : IVideoFrameWriter
{
    private bool _completed;
    private bool _disposed;

    /// <inheritdoc />
    public long FramesWritten => inner.FramesWritten;

    /// <inheritdoc />
    public ValueTask WriteAsync(ImageInput frame, CancellationToken cancellationToken = default) => inner.WriteAsync(frame, cancellationToken);

    /// <inheritdoc />
    public async Task CompleteAsync(CancellationToken cancellationToken = default)
    {
        await inner.CompleteAsync(cancellationToken).ConfigureAwait(false);
        await file.FlushAsync(cancellationToken).ConfigureAwait(false);
        _completed = true;
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        try
        {
            await inner.DisposeAsync().ConfigureAwait(false);
        }
        finally
        {
            await file.DisposeAsync().ConfigureAwait(false);
            if (!_completed)
            {
                File.Delete(path);
            }
        }
    }
}
