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
}
