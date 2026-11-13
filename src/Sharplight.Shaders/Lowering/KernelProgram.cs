using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>Where a buffer lives / how a backend should allocate it.</summary>
public enum BufferKind
{
    /// <summary>Model parameter; uploaded once, device local, read-only.</summary>
    Weight,
    /// <summary>Persistent per-sequence state; device local, survives executions.</summary>
    State,
    /// <summary>Transient activation arena; device local.</summary>
    Activations,
    /// <summary>Host-written input (tokens, positions, cross-device activations); host visible.</summary>
    Input,
    /// <summary>Host-read output; host visible (cached).</summary>
    Output,
    /// <summary>Runtime scalar block (uint[]), host visible, written before each execution.</summary>
    Runtime,
}

/// <summary>A logical buffer of a <see cref="KernelProgram"/>. Backends map it to device memory.</summary>
public sealed class BufferDecl
{
    public required int Id { get; init; }
    public required BufferKind Kind { get; init; }
    public required string Name { get; init; }
    public long Bytes { get; set; }
    public WeightInfo? Weight { get; init; }
    public StateDefinition? State { get; init; }
    public byte[]? InitialData { get; init; }
    /// <summary>Layout transform applied to the weight bytes on upload (see <see cref="WeightRepacker"/>).</summary>
    public WeightRepack Repack { get; set; }
    /// <summary>Logical elements per row (K) for <see cref="Repack"/>.</summary>
    public long RepackRowElements { get; set; }
    public override string ToString() => $"#{Id} {Kind} {Name} ({Bytes} B{(Repack != WeightRepack.None ? ", " + Repack : "")})";
}

/// <summary>Device-friendly weight layouts the lowering may choose (same byte size as the source format).</summary>
public enum WeightRepack
{
    None,
    /// <summary>
    /// Q8_0 rows → [K/32 f16 scales][K int8 quants]: quants become 8-byte aligned so kernels use wide loads.
    /// Requires K % 128 == 0. Kernels see the type <see cref="ShaderTypes.Q8_0R"/>.
    /// </summary>
    Q8_0Split,
}

/// <summary>Internal data types for repacked weights.</summary>
public static class ShaderTypes
{
    /// <summary>Q8_0 in <see cref="WeightRepack.Q8_0Split"/> layout (same 34 bytes per 32 values).</summary>
    public static readonly DataType Q8_0R = DataType.Register("q8_0r", DataTypeKind.Quantized, 32, 34);
}

/// <summary>Host-side weight layout transforms, applied row by row during upload (backend-neutral).</summary>
public static class WeightRepacker
{
    public static long RowBytes(WeightRepack r, long k) => r switch
    {
        WeightRepack.Q8_0Split => k / 32 * 34,
        _ => throw new ArgumentOutOfRangeException(nameof(r)),
    };

    /// <summary>Transforms whole rows from <paramref name="src"/> into <paramref name="dst"/> (same length).</summary>
    public static void Apply(WeightRepack r, long k, ReadOnlySpan<byte> src, Span<byte> dst)
    {
        long rowBytes = RowBytes(r, k);
        if (src.Length != dst.Length || src.Length % rowBytes != 0)
        {
            throw new ArgumentException("Repack needs whole rows.");
        }

        int blocks = (int)(k / 32);
        for (int row = 0; row < src.Length / rowBytes; row++)
        {
            var s = src.Slice((int)(row * rowBytes), (int)rowBytes);
            var d = dst.Slice((int)(row * rowBytes), (int)rowBytes);
            var scales = d[..(blocks * 2)];
            var quants = d[(blocks * 2)..];
            for (int b = 0; b < blocks; b++)
            {
                var blk = s.Slice(b * 34, 34);
                scales[b * 2] = blk[0];
                scales[b * 2 + 1] = blk[1];
                blk[2..].CopyTo(quants.Slice(b * 32, 32));
            }
        }
    }
}

/// <summary>Strided view of a tensor inside a buffer. Strides and shape are in elements; the outermost
/// dimension may be symbolic (<see cref="Symbol"/>), in which case Shape[0] holds its upper bound.</summary>
public sealed record TensorRef(BufferDecl Buffer, long ByteOffset, DataType Type, long[] Shape, long[] Strides, string? Symbol)
{
    public int Rank => Shape.Length;

    public bool IsDense
    {
        get
        {
            long s = 1;
            for (int i = Rank - 1; i >= 0; i--)
            {
                if (Shape[i] != 1 && Strides[i] != s)
                {
                    return false;
                }

                s *= Shape[i];
            }
            return true;
        }
    }

    public long MaxElements => Shape.Aggregate(1L, (a, b) => a * b);
    public long MaxBytes => Type.BlockSize == 1 ? MaxSpanElements * Type.BlockBytes : Type.ByteSize(MaxElements);

    /// <summary>Elements between the first and one-past-last addressed element (for hazard tracking).</summary>
    public long MaxSpanElements
    {
        get
        {
            long last = 0;
            for (int i = 0; i < Rank; i++)
            {
                last += (Shape[i] - 1) * Strides[i];
            }

            return last + 1;
        }
    }

    public static long[] DenseStrides(long[] shape)
    {
        var s = new long[shape.Length];
        long acc = 1;
        for (int i = shape.Length - 1; i >= 0; i--) { s[i] = acc; acc *= shape[i]; }
        return s;
    }
}

/// <summary>Argument of a kernel launch. Packed in declaration order: pointers (8 bytes) then 32-bit scalars.</summary>
public abstract record KernelArg;
/// <summary>Device address of buffer + byte offset.</summary>
public sealed record PointerArg(BufferDecl Buffer, long ByteOffset) : KernelArg;
public sealed record UIntArg(uint Value) : KernelArg;
public sealed record FloatArg(float Value) : KernelArg;
/// <summary>Value of a bound symbol (e.g. T) at record time.</summary>
public sealed record SymbolArg(string Symbol, long Multiplier = 1) : KernelArg;

/// <summary>Binding of symbolic dimensions for one recorded execution.</summary>
public sealed class Bindings(IReadOnlyDictionary<string, long> symbols)
{
    public long this[string symbol] => symbols.TryGetValue(symbol, out var v) ? v : throw new KeyNotFoundException($"Symbol '{symbol}' not bound.");
    public long Get(string? symbol, long fallback) => symbol is null ? fallback : this[symbol];
    public IReadOnlyDictionary<string, long> Values => symbols;
}

public readonly record struct Grid(uint X, uint Y = 1, uint Z = 1)
{
    public static uint Div(long n, long d) => (uint)Math.Max(1, (n + d - 1) / d);
}

/// <summary>One kernel choice for a launch (e.g. matvec vs gemm, selected by token count).</summary>
public sealed record KernelVariant(Func<Bindings, bool> When, CompiledKernel Kernel, Func<Bindings, Grid> Grid);

/// <summary>Byte range of a buffer touched by a launch (used to place barriers).</summary>
public readonly record struct BufferRange(BufferDecl Buffer, long Start, long End)
{
    public bool Overlaps(BufferRange o) => Buffer == o.Buffer && Start < o.End && o.Start < End;
}

public sealed class KernelLaunch
{
    public required string Name { get; init; }
    public required IReadOnlyList<KernelVariant> Variants { get; init; }
    public required IReadOnlyList<KernelArg> Args { get; init; }
    public required IReadOnlyList<BufferRange> Reads { get; init; }
    public required IReadOnlyList<BufferRange> Writes { get; init; }
    /// <summary>Graph node(s) implemented by this launch (for debugging / profiling).</summary>
    public IReadOnlyList<Node> Nodes { get; init; } = [];

    /// <summary>The first variant whose condition holds, or null: the launch is skipped for these bindings.</summary>
    public KernelVariant? Select(Bindings b) => Variants.FirstOrDefault(v => v.When(b));
}

/// <summary>
/// Backend-neutral executable form of a graph segment: logical buffers, kernel launches in order
/// (with hazard info), and where segment inputs/outputs/states live. Vulkan/CUDA/Metal providers
/// only allocate buffers, create pipelines and replay the launches.
/// </summary>
public sealed class KernelProgram
{
    public required IReadOnlyList<BufferDecl> Buffers { get; init; }
    public required IReadOnlyList<KernelLaunch> Launches { get; init; }
    /// <summary>Launches that reset all states (zero fill), run on ResetState.</summary>
    public required IReadOnlyList<KernelLaunch> ResetLaunches { get; init; }
    public required IReadOnlyDictionary<Value, TensorRef> InputRefs { get; init; }
    public required IReadOnlyDictionary<Value, TensorRef> OutputRefs { get; init; }
    public required BufferDecl RuntimeBuffer { get; init; }
    /// <summary>Slot index in the runtime buffer for each runtime scalar.</summary>
    public required IReadOnlyDictionary<string, int> ScalarSlots { get; init; }
    public required IReadOnlyList<string> Symbols { get; init; }
}
