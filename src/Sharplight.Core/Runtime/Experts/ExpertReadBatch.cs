namespace Sharplight.Runtime.Experts;

/// <summary>
/// The reads of one <see cref="IExpertReader.Submit"/> call: complete when every read finished, failed when one of them did. Readers report
/// each read with <see cref="CompleteRead"/> or <see cref="FailRead"/>; consumers wait with <see cref="Wait"/>.
/// </summary>
public sealed class ExpertReadBatch
{
    private readonly ManualResetEventSlim _done = new(false);
    private int _remaining;
    private long _bytes;
    private Exception? _error;

    /// <summary>Creates a batch of <paramref name="reads"/> reads (a batch of none is complete).</summary>
    /// <param name="reads">Reads the reader will report.</param>
    public ExpertReadBatch(int reads)
    {
        _remaining = reads;
        if (reads == 0)
        {
            _done.Set();
        }
    }

    /// <summary>A batch with nothing to read.</summary>
    public static ExpertReadBatch Completed => new(0);

    /// <summary>Whether every read finished (successfully or not).</summary>
    public bool IsCompleted => _done.IsSet;

    /// <summary>Whether a read failed (<see cref="Wait"/> throws).</summary>
    public bool IsFaulted => Volatile.Read(ref _error) is not null;

    /// <summary>Bytes read so far.</summary>
    public long BytesRead => Interlocked.Read(ref _bytes);

    /// <summary>Reports a read that finished with <paramref name="bytes"/> bytes.</summary>
    /// <param name="bytes">Bytes read.</param>
    public void CompleteRead(long bytes)
    {
        Interlocked.Add(ref _bytes, bytes);
        Finish();
    }

    /// <summary>Reports a read that failed; <see cref="Wait"/> throws the first such error.</summary>
    /// <param name="error">What went wrong.</param>
    public void FailRead(Exception error)
    {
        Interlocked.CompareExchange(ref _error, error, null);
        Finish();
    }

    /// <summary>Blocks until every read finished.</summary>
    /// <exception cref="IOException">A read failed (the first error is the inner exception).</exception>
    public void Wait()
    {
        _done.Wait();
        if (_error is { } error)
        {
            throw new IOException($"Reading routed experts from disk failed: {error.Message}", error);
        }
    }

    private void Finish()
    {
        if (Interlocked.Decrement(ref _remaining) == 0)
        {
            _done.Set();
        }
    }
}
