namespace Sharplight.Runtime;

/// <summary>A buffer of a <see cref="SharedArena{T}"/> and the count of its holders; the last <see cref="Release"/> frees it.</summary>
/// <typeparam name="T">The provider's buffer handle.</typeparam>
public sealed class SharedArenaBuffer<T> where T : class
{
    private readonly Action<T> _free;
    private readonly Lock _lock;
    private int _holders;

    internal SharedArenaBuffer(T buffer, long bytes, Action<T> free, Lock arenaLock)
    {
        Buffer = buffer;
        Bytes = bytes;
        _free = free;
        _lock = arenaLock;
    }

    /// <summary>The provider's buffer.</summary>
    public T Buffer { get; }

    /// <summary>Size of the buffer in bytes.</summary>
    public long Bytes { get; }

    /// <summary>Whether the last holder released the buffer (the arena then allocates a new one).</summary>
    internal bool IsFreed { get; private set; }

    /// <summary>Takes a hold (under the arena's lock).</summary>
    internal void Hold() => _holders++;

    /// <summary>Gives up one hold; the last one frees the buffer.</summary>
    public void Release()
    {
        lock (_lock)
        {
            if (--_holders > 0 || IsFreed)
            {
                return;
            }

            IsFreed = true;
        }

        _free(Buffer);
    }
}
