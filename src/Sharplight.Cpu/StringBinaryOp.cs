using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Broadcasting concatenation or equality of two dense str tensors (see <c>StandardOps.StringBinary</c>).</summary>
internal sealed unsafe class StringBinaryOp(TRef a, TRef b, TRef y, bool equal) : CpuOp
{
    public override string Name => equal ? "StringEqual" : "StringConcat";

    public override void Run(RunParameters p)
    {
        int rank = y.Rank;
        var outDims = new long[rank];
        for (int d = 0; d < rank; d++)
        {
            outDims[d] = y.Dim(d, p);
        }

        var aStrides = BroadcastStrides(a, rank, p);
        var bStrides = BroadcastStrides(b, rank, p);
        long total = y.Elements(p);
        var index = new long[rank];
        var ah = (int*)a.P;
        var bh = (int*)b.P;
        for (long i = 0; i < total; i++)
        {
            long ai = 0;
            long bi = 0;
            for (int d = 0; d < rank; d++)
            {
                ai += index[d] * aStrides[d];
                bi += index[d] * bStrides[d];
            }

            // Strings are interned, so equal handles mean equal strings.
            if (equal)
            {
                ((float*)y.P)[i] = ah[ai] == bh[bi] ? 1f : 0f;
            }
            else
            {
                ((int*)y.P)[i] = p.Strings.Add(p.Strings[ah[ai]] + p.Strings[bh[bi]]);
            }

            for (int d = rank - 1; d >= 0 && ++index[d] == outDims[d]; d--)
            {
                index[d] = 0;
            }
        }
    }

    /// <summary>Element strides of a dense operand aligned to the output's trailing dims; 0 along broadcast dims.</summary>
    private static long[] BroadcastStrides(TRef t, int rank, RunParameters p)
    {
        var strides = new long[rank];
        long stride = 1;
        for (int d = t.Rank - 1; d >= 0; d--)
        {
            long extent = t.Dim(d, p);
            strides[rank - t.Rank + d] = extent == 1 ? 0 : stride;
            stride *= extent;
        }

        return strides;
    }
}
