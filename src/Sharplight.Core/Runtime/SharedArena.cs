namespace Sharplight.Runtime;

/// <summary>
/// The activation arena that the segments of one compiled model share on one device: the segments run one after another and each
/// finishes before the next starts (<see cref="CompiledModel.Run"/>), so their transient activations can live in the same memory. A model
/// split into many segments per device (the routed experts of some layers on the CPU, the rest on the GPU) then holds one arena per device
/// instead of one per segment. Providers keep one instance per device and <see cref="CompileOptions.StateScope"/>.
/// </summary>
/// <typeparam name="T">The provider's buffer handle.</typeparam>
/// <remarks>
/// A segment that needs more than the current buffer holds gets a new, larger buffer, which later segments share; the earlier
/// segments keep theirs. Every holder calls <see cref="SharedArenaBuffer{T}.Release"/> once; the last one frees the buffer. Thread-safe.
/// <c>SHARPLIGHT_SHARED_ARENA=0</c> turns sharing off (<see cref="Enabled"/>).
/// </remarks>
public sealed class SharedArena<T> where T : class
{
    private readonly Lock _lock = new();
    private SharedArenaBuffer<T>? _current;

    /// <summary>Whether providers share activation arenas (default true; <c>SHARPLIGHT_SHARED_ARENA=0</c> turns it off).</summary>
    public static bool Enabled { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_SHARED_ARENA") is not ("0" or "false");

    /// <summary>Takes a hold on a buffer of at least <paramref name="bytes"/>, allocating it with <paramref name="allocate"/> when the current one is smaller.</summary>
    /// <param name="bytes">Bytes the segment's arena needs.</param>
    /// <param name="allocate">Allocates a buffer of the given size (called under the lock).</param>
    /// <param name="free">Frees a buffer once its last holder released it.</param>
    /// <param name="allocated">Whether a new buffer was allocated for this request.</param>
    public SharedArenaBuffer<T> Acquire(long bytes, Func<long, T> allocate, Action<T> free, out bool allocated)
    {
        ArgumentNullException.ThrowIfNull(allocate);
        ArgumentNullException.ThrowIfNull(free);
        lock (_lock)
        {
            allocated = _current is null || _current.IsFreed || _current.Bytes < bytes;
            if (allocated)
            {
                _current = new SharedArenaBuffer<T>(allocate(bytes), bytes, free, _lock);
            }

            _current!.Hold();
            return _current;
        }
    }
}
