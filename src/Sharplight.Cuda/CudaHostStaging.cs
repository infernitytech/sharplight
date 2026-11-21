using Sharplight.Graph;

namespace Sharplight.Cuda;

/// <summary>
/// Page-locked host buffers of a segment's small inputs and outputs, one per value at its largest size: inputs are copied into them and
/// uploaded asynchronously on the segment's stream before its launches, outputs are downloaded into them asynchronously after the launches,
/// so a step pays one wait instead of a synchronous copy per value (decode steps that hand a layer to the CPU and back exchange
/// four to six such values per segment). Values larger than <see cref="MaxBytes"/>, or when no page-locked memory is left, are not staged.
/// Not thread-safe: the segment's callers serialize its inputs, executions and outputs.
/// </summary>
internal sealed unsafe class CudaHostStaging : IDisposable
{
    /// <summary>Largest value staged, in bytes (the logits of one row of a 256k vocabulary fit).</summary>
    public const long MaxBytes = 1L << 20;

    private readonly CudaDevice _dev;
    private readonly Dictionary<Value, (nint Memory, long Bytes)> _buffers = [];
    private readonly HashSet<Value> _refused = [];

    public CudaHostStaging(CudaDevice dev)
    {
        _dev = dev;
    }

    /// <summary>The staging buffer of <paramref name="value"/> (allocated at <paramref name="capacity"/> bytes on first use), or null when it is not staged.</summary>
    /// <param name="value">The input or output.</param>
    /// <param name="capacity">The value's largest size in bytes.</param>
    public byte* Get(Value value, long capacity)
    {
        if (_buffers.TryGetValue(value, out var buffer))
        {
            return (byte*)buffer.Memory;
        }

        if (capacity <= 0 || capacity > MaxBytes || _refused.Contains(value))
        {
            return null;
        }

        byte* memory = _dev.TryAllocatePinned(capacity);
        if (memory == null)
        {
            _refused.Add(value);
            return null;
        }

        _buffers[value] = ((nint)memory, capacity);
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
