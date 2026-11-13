using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>
/// A graph segment compiled for the CPU: every value is a (buffer, offset, dims, strides) view, activation
/// buffers are packed into one arena by liveness, weights are used in place (zero-copy when memory mapped).
/// </summary>
/// <remarks>
/// The class is split by concern: <c>Values</c> (buffers and value references), <c>Views</c> (zero-copy view
/// arithmetic), <c>Arena</c> (liveness-based packing), <c>Lowering</c> (node dispatch) and one <c>Lower*</c> file per
/// op category.
/// </remarks>
internal sealed unsafe partial class CpuCompiledSegment : ICompiledSegment
{
    private readonly GraphSegment _segment;
    private readonly CompileOptions _options;
    private readonly Dictionary<Value, TRef> _refs = [];
    private readonly Dictionary<Value, TRef> _inputs = [];
    private readonly Dictionary<Value, TRef> _f32Cache = [];
    private readonly Dictionary<WeightInfo, Buf> _weightBufs = [];
    private readonly List<(CpuOp Op, TRef[] Touches)> _steps = [];
    private readonly List<Buf> _bufs = [];
    private readonly List<TRef> _allRefs = [];
    private readonly List<nint> _owned = [];
    private readonly List<(Buf State, long Bytes)> _stateBufs = [];
    private readonly Dictionary<string, long> _symbolBounds = new(StringComparer.Ordinal);
    private byte* _arena;
    private long _arenaBytes;
    private long _ownedBytes;
    private long _weightBytes;
    private RunParameters? _last;
    private bool _disposed;

    internal CpuCompiledSegment(GraphSegment segment, CompileOptions options, CpuDevice device)
    {
        _segment = segment;
        _options = options;
        try
        {
            foreach (var input in segment.Inputs)
            {
                Ref(input);
            }

            foreach (var node in segment.Nodes)
            {
                Lower(node, device);
            }

            foreach (var output in segment.Outputs)
            {
                if (_refs.TryGetValue(output, out var r))
                {
                    r.Buf.Pinned = true;
                }
            }

            AllocateArena();
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public GraphSegment Segment => _segment;

    /// <inheritdoc />
    public long DeviceMemoryBytes => _arenaBytes + _ownedBytes + _weightBytes;

    /// <summary>Bytes of the shared activation arena.</summary>
    public long ArenaBytes => _arenaBytes;

    /// <inheritdoc />
    public void SetInput(Value value, ReadOnlySpan<byte> data)
    {
        if (!_inputs.TryGetValue(value, out var r))
        {
            throw new ArgumentException($"{value} is not an input of this segment.");
        }

        if (data.Length > r.Buf.Bytes)
        {
            throw new ArgumentException($"Input {value.Name}: {data.Length} bytes exceed the compiled capacity of {r.Buf.Bytes} bytes (MaxTokens too small?).");
        }

        data.CopyTo(new Span<byte>(r.P, data.Length));
    }

    /// <inheritdoc />
    public void Execute(RunParameters parameters)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        foreach (var (symbol, bound) in _symbolBounds)
        {
            if (SymbolicExpression.TryEvaluate(symbol, parameters.Symbols, out var v) && v > bound)
            {
                throw new InvalidOperationException($"Symbol '{symbol}' = {v} exceeds the compiled bound {bound}.");
            }
        }

        _last = parameters;
        foreach (var (op, _) in _steps)
        {
            op.Run(parameters);
        }
    }

    /// <inheritdoc />
    public void GetOutput(Value value, Span<byte> destination)
    {
        if (!_refs.TryGetValue(value, out var r))
        {
            throw new ArgumentException($"{value} is not a value of this segment.");
        }

        var p = _last ?? throw new InvalidOperationException("GetOutput called before Execute.");
        long elems = r.Elements(p);
        long bytes = elems * r.ElemSize;
        if (destination.Length < bytes)
        {
            throw new ArgumentException($"Destination too small: need {bytes} bytes, got {destination.Length}.");
        }

        if (elems == 0)
        {
            return;
        }

        var dims = new long[r.Rank];
        for (int i = 0; i < dims.Length; i++)
        {
            dims[i] = r.Dim(i, p);
        }

        fixed (byte* d = destination)
        {
            Mem.CopyStrided(r.P, r.Strides, d, TRef.RowMajor(dims), r.Dims, r.Rank == 0 ? 1 : dims[0], r.ElemSize);
        }
    }

    /// <summary>Overwrites a state tensor with host data (f32 states; used by tests and for state restore).</summary>
    public void TestSetState(Value state, float[] data)
    {
        var r = _refs[state];
        if (r.Type != DataType.F32)
        {
            throw new NotSupportedException("f32 states only");
        }

        data.AsSpan().CopyTo(new Span<float>(r.P, data.Length));
    }

    /// <summary>Reads back an f32 state tensor.</summary>
    public float[] TestGetState(Value state)
    {
        var r = _refs[state];
        var result = new float[r.MaxElements];
        new Span<float>(r.P, result.Length).CopyTo(result);
        return result;
    }

    /// <inheritdoc />
    public void ResetState()
    {
        foreach (var (buf, bytes) in _stateBufs)
        {
            System.Runtime.InteropServices.NativeMemory.Clear(buf.Ptr, (nuint)bytes);
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (var (op, _) in _steps)
        {
            op.Dispose();
        }

        Mem.Free(_arena);
        _arena = null;
        foreach (var p in _owned)
        {
            Mem.Free((byte*)p);
        }

        _owned.Clear();
        _bufs.Clear();
        _steps.Clear();
    }
}
