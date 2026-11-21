using Sharplight.Graph;

namespace Sharplight.Cuda;

/// <summary>
/// Page-locked host buffers of a segment's small inputs and outputs, one per value (at most <see cref="MaxBytes"/>): inputs are copied into them and
/// uploaded asynchronously on the segment's stream before its launches, outputs are downloaded into them asynchronously after the launches,
/// so a step pays one wait instead of a synchronous copy per value (decode steps that hand a layer to the CPU and back exchange
/// four to six such values per segment). Transfers larger than <see cref="MaxBytes"/>, and all of them when no page-locked memory is left, are not staged.
/// Not thread-safe: the segment's callers serialize its inputs, executions and outputs.
/// </summary>
internal sealed unsafe class CudaHostStaging : IDisposable
{
    /// <summary>Largest staging buffer, in bytes (the logits of one row of a 256k vocabulary fit).</summary>
    public const long MaxBytes = 1L << 20;

    private readonly CudaDevice _dev;
    private readonly Dictionary<Value, (nint Memory, long Bytes)> _buffers = [];
    private readonly HashSet<Value> _refused = [];

    public CudaHostStaging(CudaDevice dev)
    {
        _dev = dev;
    }

    /// <summary>
    /// The staging buffer of <paramref name="value"/>, allocated on first use at its largest size or <see cref="MaxBytes"/>, whichever is
    /// smaller (a decode step's rows fit where a prefill pass's do not), or null when it has none.
    /// </summary>
    /// <param name="value">The input or output.</param>
    /// <param name="largest">The value's largest size in bytes.</param>
    /// <param name="capacity">The buffer's size: a transfer of more bytes does not use it.</param>
    public byte* Get(Value value, long largest, out long capacity)
    {
        if (_buffers.TryGetValue(value, out var buffer))
        {
            capacity = buffer.Bytes;
            return (byte*)buffer.Memory;
        }

        capacity = 0;
        long bytes = Math.Min(largest, MaxBytes);
        if (bytes <= 0 || _refused.Contains(value))
        {
            return null;
        }

        byte* memory = _dev.TryAllocatePinned(bytes);
        if (memory == null)
        {
            _refused.Add(value);
            return null;
        }

        _buffers[value] = ((nint)memory, bytes);
        capacity = bytes;
        return memory;
    }

    public void Dispose()
    {
        foreach (var (memory, bytes) in _buffers.Values)
        {
            _dev.FreeHost((byte*)memory, bytes, pinned: true);
        }

        _buffers.Clear();
    }
}
