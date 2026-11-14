namespace Sharplight.Runtime;

/// <summary>
/// A compiled segment's hold on a shared weight buffer (<see cref="SharedWeights.Acquire{T}"/>). The first holder of a buffer
/// allocated it and fills it (<see cref="MustFill"/>), then calls <see cref="Filled"/>; later holders call
/// <see cref="WaitFilled"/> before their first use. Disposing the last lease of a buffer frees it.
/// </summary>
/// <typeparam name="T">The provider's buffer handle.</typeparam>
public sealed class SharedWeightLease<T> : IDisposable where T : class
{
    private readonly SharedWeights _owner;
    private readonly SharedWeightEntry _entry;
    private bool _disposed;

    internal SharedWeightLease(SharedWeights owner, SharedWeightEntry entry, bool mustFill)
    {
        _owner = owner;
        _entry = entry;
        MustFill = mustFill;
    }

    /// <summary>The device buffer.</summary>
    public T Buffer => (T)_entry.Buffer;

    /// <summary>True for the holder that allocated the buffer: it must write the weight's bytes and then call <see cref="Filled"/>.</summary>
    public bool MustFill { get; }

    /// <summary>The filling holder's data is in place (on the device, visible to later submissions).</summary>
    /// <exception cref="InvalidOperationException">This lease did not allocate the buffer.</exception>
    public void Filled()
    {
        if (!MustFill)
        {
            throw new InvalidOperationException("Only the lease that allocated a shared weight fills it.");
        }

        _entry.MarkFilled(failed: false);
    }

    /// <summary>Blocks until the buffer's data is in place (immediately for the filling holder once it called <see cref="Filled"/>).</summary>
    /// <exception cref="InvalidOperationException">The filling holder failed before the data was in place.</exception>
    public void WaitFilled() => WaitFilled(CancellationToken.None);

    /// <summary>Blocks until the buffer's data is in place, or until <paramref name="ct"/> is cancelled.</summary>
    /// <param name="ct">Stops waiting (the waiting compilation is being cancelled).</param>
    /// <exception cref="InvalidOperationException">The filling holder failed (or was cancelled) before the data was in place.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="ct"/> was cancelled first.</exception>
    public void WaitFilled(CancellationToken ct)
    {
        _entry.Ready.Wait(ct);
        if (_entry.Failed)
        {
            throw new InvalidOperationException("The compilation that uploads this shared weight failed.");
        }
    }

    /// <summary>Releases the hold; the last one frees the buffer. A filling holder that never called <see cref="Filled"/> marks it failed.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _owner.Release(_entry, MustFill);
    }
}
