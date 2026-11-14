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
internal sealed unsafe partial class CpuCompiledSegment : ICompiledSegment, IPartialOutputSegment
{
    private readonly GraphSegment _segment;
    private readonly CompileOptions _options;
    private readonly IReadOnlySet<string> _features;
    private readonly Dictionary<Value, TRef> _refs = [];
    private readonly Dictionary<Value, TRef> _inputs = [];
    private readonly Dictionary<Value, TRef> _f32Cache = [];
    private readonly Dictionary<WeightInfo, Buf> _weightBufs = [];
    private readonly List<(CpuOp Op, TRef[] Touches)> _steps = [];
    private readonly List<Buf> _bufs = [];
    private readonly List<TRef> _allRefs = [];
    private readonly List<nint> _owned = [];
    /// <summary>Weight memory held through <see cref="CompileOptions.SharedWeights"/> (freed with the last holder).</summary>
    private readonly List<IDisposable> _leases = [];
    private readonly CpuDevice _device;
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
        _device = device;
        _features = options.FeaturesFor(device);
        try
        {
            foreach (var input in segment.Inputs)
            {
                Ref(input);
            }

            IndexConsumers();
            var progress = new LoadProgressCounter(options.Progress, LoadStage.Compiling, segment.Nodes.Count, device.Info.Name);
            foreach (var node in segment.Nodes)
            {
                options.CancellationToken.ThrowIfCancellationRequested();
                if (!_absorbed.Contains(node))
                {
                    Lower(node, device);
                }

                progress.Add();
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
        if (CpuProfiler.Enabled)
        {
            foreach (var (op, _) in _steps)
            {
                CpuProfiler.Run(op, parameters);
            }

            return;
        }

        foreach (var (op, _) in _steps)
        {
            op.Run(parameters);
        }
    }

    /// <inheritdoc />
    public void GetOutput(Value value, long byteOffset, Span<byte> destination)
    {
        if (!_refs.TryGetValue(value, out var r))
        {
            throw new ArgumentException($"{value} is not a value of this segment.");
        }

        var p = _last ?? throw new InvalidOperationException("GetOutput called before Execute.");
        var dims = new long[r.Rank];
        for (int i = 0; i < dims.Length; i++)
        {
            dims[i] = r.Dim(i, p);
        }

        long bytes = r.Elements(p) * r.ElemSize;
        if (byteOffset < 0 || byteOffset + destination.Length > bytes)
        {
            throw new ArgumentOutOfRangeException(nameof(byteOffset), $"Range [{byteOffset}, {byteOffset + destination.Length}) outside the {bytes}-byte output.");
        }

        var dense = TRef.RowMajor(dims);
        bool contiguous = true;
        for (int i = 0; i < dims.Length; i++)
        {
            contiguous &= dims[i] <= 1 || r.Strides[i] == dense[i];
        }

        if (contiguous)
        {
            new ReadOnlySpan<byte>(r.P + byteOffset, destination.Length).CopyTo(destination);
            return;
        }

        var whole = new byte[bytes];
        GetOutput(value, whole);
        whole.AsSpan(checked((int)byteOffset), destination.Length).CopyTo(destination);
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

    /// <summary>Overwrites a state tensor with host data (f32, f16 or bf16 states, rounded to nearest; used by tests and for state restore).</summary>
    public void TestSetState(Value state, float[] data)
    {
        var r = _refs[state];
        if (r.Type == DataType.F32)
        {
            data.AsSpan().CopyTo(new Span<float>(r.P, data.Length));
        }
        else if (r.Type == DataType.F16 || r.Type == DataType.BF16)
        {
            var d = (ushort*)r.P;
            for (int i = 0; i < data.Length; i++)
            {
                d[i] = r.Type == DataType.F16 ? Simd.FloatToHalf(data[i]) : Simd.FloatToBFloat(data[i]);
            }
        }
        else
        {
            throw new NotSupportedException("f32, f16 and bf16 states only");
        }
    }

    /// <summary>Reads back an f32, f16 or bf16 state tensor as f32.</summary>
    public float[] TestGetState(Value state)
    {
        var r = _refs[state];
        var result = new float[r.MaxElements];
        if (r.Type == DataType.F16 || r.Type == DataType.BF16)
        {
            var s = (ushort*)r.P;
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = r.Type == DataType.F16 ? Simd.HalfToFloat(s[i]) : Simd.BFloatToFloat(s[i]);
            }

            return result;
        }

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
        foreach (var lease in _leases)
        {
            lease.Dispose();
        }

        _leases.Clear();
        _bufs.Clear();
        _steps.Clear();
    }
}
