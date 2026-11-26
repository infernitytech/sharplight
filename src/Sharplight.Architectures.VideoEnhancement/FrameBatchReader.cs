using Sharplight.Vision;

namespace Sharplight.Architectures.VideoEnhancement;

/// <summary>
/// Reads the frames of a request in batches on the enhancer's compute thread, one batch ahead: while a batch is being enhanced the next one
/// is read (decoded) on the thread pool, so a frame source never holds the compute back and never runs more than one batch ahead.
/// Not thread safe; one instance per request.
/// </summary>
internal sealed class FrameBatchReader : IDisposable
{
    private readonly IAsyncEnumerator<ImageInput> _source;
    private readonly CancellationToken _cancellationToken;
    private Task<List<ImageInput>>? _ahead;
    private List<ImageInput>? _head;
    private bool _ended;

    /// <summary>Starts reading <paramref name="frames"/>.</summary>
    /// <param name="frames">The frames in order.</param>
    /// <param name="cancellationToken">Stops the reading.</param>
    public FrameBatchReader(IAsyncEnumerable<ImageInput> frames, CancellationToken cancellationToken)
    {
        _source = frames.GetAsyncEnumerator(cancellationToken);
        _cancellationToken = cancellationToken;
    }

    /// <summary>The first frame without taking it (the next <see cref="Take"/> starts with it), or <see langword="null"/> for an empty source.</summary>
    public ImageInput? PeekFirst()
    {
        _head ??= ReadAsync(1, []).GetAwaiter().GetResult();
        return _head.Count > 0 ? _head[0] : null;
    }

    /// <summary>The next <paramref name="count"/> frames (fewer at the end of the source, none after it); then starts reading the batch after them.</summary>
    /// <param name="count">The frames wanted (positive; every call of a request asks for the same number).</param>
    public List<ImageInput> Take(int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        var frames = _ahead?.GetAwaiter().GetResult() ?? _head ?? [];
        _ahead = null;
        _head = null;
        if (frames.Count < count && !_ended)
        {
            frames = ReadAsync(count, frames).GetAwaiter().GetResult();
        }

        if (!_ended && count < int.MaxValue)
        {
            _ahead = Task.Run(() => ReadAsync(count, []), CancellationToken.None);
        }

        return frames;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        try
        {
            _ahead?.GetAwaiter().GetResult();
        }
        catch (Exception error) when (error is OperationCanceledException or InvalidOperationException or IOException or InvalidDataException)
        {
            // The request is over; a read that failed or was cancelled after the last batch it served has nothing left to report.
        }

        _source.DisposeAsync().AsTask().GetAwaiter().GetResult();
    }

    /// <summary>Adds frames to <paramref name="frames"/> until it holds <paramref name="count"/> or the source ends.</summary>
    private async Task<List<ImageInput>> ReadAsync(int count, List<ImageInput> frames)
    {
        while (frames.Count < count && !_ended)
        {
            _cancellationToken.ThrowIfCancellationRequested();
            if (await _source.MoveNextAsync().ConfigureAwait(false))
            {
                frames.Add(_source.Current);
            }
            else
            {
                _ended = true;
            }
        }

        return frames;
    }
}
