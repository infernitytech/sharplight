using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>Shared helpers of the general-purpose data-movement ops (dense inputs; element-size generic copies).</summary>
internal static unsafe class OpHelpers
{
    public static void Cp(byte* d, byte* s, int es)
    {
        switch (es)
        {
            case 4:
                *(int*)d = *(int*)s;
                break;
            case 8:
                *(long*)d = *(long*)s;
                break;
            case 2:
                *(short*)d = *(short*)s;
                break;
            default:
                *d = *s;
                break;
        }
    }

    /// <summary>Reads element <paramref name="i"/> of a dense i32/i64 index tensor.</summary>
    public static long Idx(TRef t, long i) => t.Type == DataType.I64 ? ((long*)t.P)[i] : ((int*)t.P)[i];

    public static long[] Dims(TRef t, RunParameters p)
    {
        var d = new long[t.Rank];
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = t.Dim(i, p);
        }

        return d;
    }

    public static long Prod(long[] d, int from, int to)
    {
        long n = 1;
        for (int i = from; i < to; i++)
        {
            n *= d[i];
        }

        return n;
    }

    public static long[] Strides(long[] d)
    {
        var s = new long[d.Length];
        long e = 1;
        for (int i = d.Length - 1; i >= 0; i--)
        {
            s[i] = e;
            e *= d[i];
        }
        return s;
    }

    public static long Wrap(long id, long n, string what)
    {
        if (id < 0)
        {
            id += n;
        }

        if ((ulong)id >= (ulong)n)
        {
            throw new ArgumentOutOfRangeException("indices", $"{what} index {id} out of range [0,{n}).");
        }

        return id;
    }

    public static (int chunks, long per) Split(long n, long work)
    {
        int c = (int)Math.Max(1, Math.Min(n, CpuParallel.Chunks(n * Math.Max(1, work), 8192)));
        return (c, (n + c - 1) / c);
    }

    /// <summary>Broadcast element strides of t aligned to the output rank (0 on broadcast axes).</summary>
    public static long[] Align(TRef t, int outRank)
    {
        var s = new long[outRank];
        for (int k = 0; k < outRank; k++)
        {
            int ik = k - (outRank - t.Rank);
            if (ik < 0)
            {
                s[k] = 0;
            }
            else if (t.Dims[ik] == 1 && !(ik == 0 && t.Sym is not null))
            {
                s[k] = 0;
            }
            else
            {
                s[k] = t.Strides[ik];
            }
        }
        return s;
    }
}
