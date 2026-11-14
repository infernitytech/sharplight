using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Sharplight.Cpu;

/// <summary>Native memory helpers: 64-byte aligned allocation and strided copies.</summary>
internal static unsafe class Mem
{
    public static byte* Alloc(long bytes, bool zero)
    {
        nuint n = (nuint)Math.Max(64, (bytes + 63) & ~63L);
        void* p = NativeMemory.AlignedAlloc(n, 64);
        if (zero)
        {
            NativeMemory.Clear(p, n);
        }

        return (byte*)p;
    }

    public static void Free(byte* p) { if (p != null)
        {
            NativeMemory.AlignedFree(p);
        }
    }

    /// <summary>N-d strided copy into a row-major (or strided) destination. Effective dims[0] is <paramref name="dim0"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void CopyStrided(byte* src, long[] srcStrides, byte* dst, long[] dstStrides, long[] dims, long dim0, int es)
    {
        int rank = dims.Length;
        if (rank == 0)
        {
            Buffer.MemoryCopy(src, dst, es, es);
            return;
        }
        Span<long> dm = stackalloc long[rank];
        dims.CopyTo(dm);
        dm[0] = dim0;
        long total = 1;
        foreach (var d in dm)
        {
            total *= d;
        }

        if (total == 0)
        {
            return;
        }

        long inner = dm[rank - 1];
        long ss = srcStrides[rank - 1];
        long ds = dstStrides[rank - 1];
        if (es == 4 && ss != 1 && ds == 1 && rank >= 2 && TryCopyTransposed(src, srcStrides, dst, dstStrides, dm))
        {
            return;
        }

        long outer = total / inner;
        Span<long> idx = stackalloc long[rank];
        idx.Clear();
        long so = 0;
        long dOff = 0;
        for (long o = 0; o < outer; o++)
        {
            byte* s = src + so * es;
            byte* d = dst + dOff * es;
            if (ss == 1 && ds == 1)
            {
                Buffer.MemoryCopy(s, d, inner * es, inner * es);
            }
            else
            {
                for (long i = 0; i < inner; i++)
                {
                    Buffer.MemoryCopy(s + i * ss * es, d + i * ds * es, es, es);
                }
            }

            for (int k = rank - 2; k >= 0; k--)
            {
                idx[k]++;
                so += srcStrides[k];
                dOff += dstStrides[k];
                if (idx[k] < dm[k])
                {
                    break;
                }

                so -= srcStrides[k] * dm[k];
                dOff -= dstStrides[k] * dm[k];
                idx[k] = 0;
            }
        }
    }

    /// <summary>Tile edge of the blocked transpose copy (a 16 x 16 tile of both sides stays in L1).</summary>
    private const int TransposeTile = 16;

    /// <summary>
    /// 4-byte copy whose destination is contiguous along the last axis and whose source is contiguous along another axis
    /// <c>a</c> (a transposed view): copies 2-D tiles of (axis a, last axis) for every index of the other axes, reading
    /// and writing whole cache lines instead of one element per line. False when no source axis has unit stride.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static bool TryCopyTransposed(byte* src, long[] srcStrides, byte* dst, long[] dstStrides, Span<long> dims)
    {
        int rank = dims.Length;
        int a = -1;
        for (int k = rank - 2; k >= 0; k--)
        {
            if (srcStrides[k] == 1 && dims[k] > 1)
            {
                a = k;
                break;
            }
        }

        if (a < 0)
        {
            return false;
        }

        long rowsA = dims[a];
        long cols = dims[rank - 1];
        long sCol = srcStrides[rank - 1];
        long dRow = dstStrides[a];
        long outer = 1;
        for (int k = 0; k < rank - 1; k++)
        {
            if (k != a)
            {
                outer *= dims[k];
            }
        }

        Span<long> idx = stackalloc long[rank];
        idx.Clear();
        long so = 0;
        long dOff = 0;
        for (long o = 0; o < outer; o++)
        {
            int* s = (int*)src + so;
            int* d = (int*)dst + dOff;
            for (long i0 = 0; i0 < rowsA; i0 += TransposeTile)
            {
                long i1 = Math.Min(rowsA, i0 + TransposeTile);
                for (long j0 = 0; j0 < cols; j0 += TransposeTile)
                {
                    long j1 = Math.Min(cols, j0 + TransposeTile);
                    for (long i = i0; i < i1; i++)
                    {
                        int* dr = d + i * dRow;
                        int* sr = s + i;
                        for (long j = j0; j < j1; j++)
                        {
                            dr[j] = sr[j * sCol];
                        }
                    }
                }
            }

            for (int k = rank - 2; k >= 0; k--)
            {
                if (k == a)
                {
                    continue;
                }

                idx[k]++;
                so += srcStrides[k];
                dOff += dstStrides[k];
                if (idx[k] < dims[k])
                {
                    break;
                }

                so -= srcStrides[k] * dims[k];
                dOff -= dstStrides[k] * dims[k];
                idx[k] = 0;
            }
        }

        return true;
    }
}
