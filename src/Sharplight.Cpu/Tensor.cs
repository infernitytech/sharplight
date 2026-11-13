using System.Runtime.InteropServices;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>A memory region (arena slice, state, weight pointer or constant).</summary>
internal sealed unsafe class Buf
{
    public long Bytes;
    public byte* Ptr;
    /// <summary>First/last step touching the buffer (for arena reuse).</summary>
    public int First = int.MaxValue, Last = -1;
    /// <summary>Lives in the shared activation arena (otherwise Ptr is already final).</summary>
    public bool InArena;
    public long ArenaOffset;
    /// <summary>Keep alive until the end of the segment.</summary>
    public bool Pinned;
    public string Name = "";
}

/// <summary>A strided view onto a <see cref="Buf"/>: element type, dims (outermost may be symbolic) and element strides.</summary>
internal sealed unsafe class TRef
{
    public required DataType Type;
    public required long[] Dims;     // Dims[0] is a placeholder (bound) when Sym != null
    public required long[] Strides;  // in elements
    public string? Sym;
    public required Buf Buf;
    public long Offset;              // in elements
    public byte* P;                  // resolved after arena allocation
    public int ElemSize => Type.BlockSize == 1 ? Type.BlockBytes : 0;
    public int Rank => Dims.Length;

    public long Dim(int i, RunParameters p) => i == 0 && Sym is not null ? p.Symbols[Sym] : Dims[i];

    public long Elements(RunParameters p)
    {
        long n = 1;
        for (int i = 0; i < Rank; i++) n *= Dim(i, p);
        return n;
    }

    public long Cols(RunParameters p) => Rank == 0 ? 1 : Dim(Rank - 1, p);

    public long Rows(RunParameters p)
    {
        long n = 1;
        for (int i = 0; i < Rank - 1; i++) n *= Dim(i, p);
        return n;
    }

    public long MaxElements
    {
        get { long n = 1; foreach (var d in Dims) n *= d; return n; }
    }

    /// <summary>Pointer to row <paramref name="row"/> (row = linear index over all but the last axis).</summary>
    public byte* RowPtr(long row)
    {
        long off = 0;
        for (int k = Rank - 2; k >= 1; k--)
        {
            long d = Dims[k];
            long q = row / d;
            off += (row - q * d) * Strides[k];
            row = q;
        }
        if (Rank >= 2) off += row * Strides[0];
        return P + off * ElemSize;
    }

    public float* F(long row) => (float*)RowPtr(row);

    public bool LastContiguous => Rank == 0 || Dims[^1] == 1 || Strides[^1] == 1;

    /// <summary>Row-major dense (ignoring size-1 dims).</summary>
    public bool IsDense
    {
        get
        {
            long expect = 1;
            for (int i = Rank - 1; i >= 0; i--)
            {
                if (Dims[i] != 1 && Strides[i] != expect) return false;
                expect *= Dims[i];
            }
            return true;
        }
    }

    public static long[] RowMajor(long[] dims)
    {
        var s = new long[dims.Length];
        long e = 1;
        for (int i = dims.Length - 1; i >= 0; i--) { s[i] = e; e *= dims[i]; }
        return s;
    }
}

/// <summary>Base class of all executable steps.</summary>
internal abstract class CpuOp
{
    public abstract string Name { get; }
    public abstract void Run(RunParameters p);
    public virtual void Dispose() { }
    public override string ToString() => Name;
}

/// <summary>Minimal parallel-for wrapper with a shared degree of parallelism.</summary>
internal static class Par
{
    static int s_threads = Math.Max(1, Environment.ProcessorCount);
    static ParallelOptions s_opts = new() { MaxDegreeOfParallelism = s_threads };

    /// <summary>Degree of parallelism used by all kernels (process-wide).</summary>
    public static int Threads
    {
        get => s_threads;
        set { s_threads = Math.Max(1, value); s_opts = new ParallelOptions { MaxDegreeOfParallelism = s_threads }; }
    }

    public static void For(int n, Action<int> body)
    {
        if (n <= 0) return;
        if (n == 1 || Threads == 1) { for (int i = 0; i < n; i++) body(i); return; }
        Parallel.For(0, n, s_opts, body);
    }

    /// <summary>Number of chunks for <paramref name="total"/> work items of which each chunk should have at least <paramref name="minChunk"/>.</summary>
    public static int Chunks(long total, long minChunk)
    {
        if (total <= minChunk) return 1;
        long c = Math.Min(Threads * 4L, total / minChunk);
        return (int)Math.Max(1, c);
    }
}

internal static unsafe class Mem
{
    public static byte* Alloc(long bytes, bool zero)
    {
        nuint n = (nuint)Math.Max(64, (bytes + 63) & ~63L);
        void* p = NativeMemory.AlignedAlloc(n, 64);
        if (zero) NativeMemory.Clear(p, n);
        return (byte*)p;
    }

    public static void Free(byte* p) { if (p != null) NativeMemory.AlignedFree(p); }

    /// <summary>N-d strided copy into a row-major (or strided) destination. Effective dims[0] is <paramref name="dim0"/>.</summary>
    public static void CopyStrided(byte* src, long[] srcStrides, byte* dst, long[] dstStrides, long[] dims, long dim0, int es)
    {
        int rank = dims.Length;
        if (rank == 0) { Buffer.MemoryCopy(src, dst, es, es); return; }
        Span<long> dm = stackalloc long[rank];
        dims.CopyTo(dm);
        dm[0] = dim0;
        long total = 1;
        foreach (var d in dm) total *= d;
        if (total == 0) return;
        long inner = dm[rank - 1];
        long ss = srcStrides[rank - 1], ds = dstStrides[rank - 1];
        long outer = total / inner;
        Span<long> idx = stackalloc long[rank];
        idx.Clear();
        long so = 0, dOff = 0;
        for (long o = 0; o < outer; o++)
        {
            byte* s = src + so * es; byte* d = dst + dOff * es;
            if (ss == 1 && ds == 1) Buffer.MemoryCopy(s, d, inner * es, inner * es);
            else
                for (long i = 0; i < inner; i++)
                    Buffer.MemoryCopy(s + i * ss * es, d + i * ds * es, es, es);
            for (int k = rank - 2; k >= 0; k--)
            {
                idx[k]++; so += srcStrides[k]; dOff += dstStrides[k];
                if (idx[k] < dm[k]) break;
                so -= srcStrides[k] * dm[k]; dOff -= dstStrides[k] * dm[k]; idx[k] = 0;
            }
        }
    }
}
