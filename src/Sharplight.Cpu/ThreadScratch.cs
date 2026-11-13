namespace Sharplight.Cpu;

/// <summary>
/// Grow-only, 64-byte aligned scratch memory per thread and slot, for kernels whose temporaries are too large for the stack
/// (packed GEMM panels, quantized activations, decoded weight rows). A buffer lives as long as its thread; slots let one
/// thread hold several buffers at once. Callers must not keep a pointer across another <see cref="Get"/> of the same slot.
/// </summary>
internal static unsafe class ThreadScratch
{
    /// <summary>The calling thread's buffer of an op's <c>Run</c> (data shared with the parallel workers).</summary>
    public const int Caller = 0;

    /// <summary>A worker's private buffer inside a parallel body.</summary>
    public const int Worker = 1;

    private const int SlotCount = 2;

    [ThreadStatic] private static nint[]? t_buffers;
    [ThreadStatic] private static long[]? t_sizes;

    /// <summary>At least <paramref name="bytes"/> bytes of the calling thread's buffer <paramref name="slot"/>; contents are undefined.</summary>
    public static byte* Get(int slot, long bytes)
    {
        var buffers = t_buffers ??= new nint[SlotCount];
        var sizes = t_sizes ??= new long[SlotCount];
        if (sizes[slot] < bytes)
        {
            Mem.Free((byte*)buffers[slot]);
            long size = Math.Max(bytes, sizes[slot] * 3 / 2);
            buffers[slot] = (nint)Mem.Alloc(size, false);
            sizes[slot] = size;
        }

        return (byte*)buffers[slot];
    }
}
