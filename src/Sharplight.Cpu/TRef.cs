using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

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

    public long Dim(int i, RunParameters p) => i == 0 && Sym is not null ? SymbolicExpression.Evaluate(Sym, p.Symbols) : Dims[i];

    public long Elements(RunParameters p)
    {
        long n = 1;
        for (int i = 0; i < Rank; i++)
        {
            n *= Dim(i, p);
        }

        return n;
    }

    public long Cols(RunParameters p) => Rank == 0 ? 1 : Dim(Rank - 1, p);

    public long Rows(RunParameters p)
    {
        long n = 1;
        for (int i = 0; i < Rank - 1; i++)
        {
            n *= Dim(i, p);
        }

        return n;
    }

    public long MaxElements
    {
        get
        {
            long n = 1;
            foreach (var d in Dims)
            {
                n *= d;
            }
            return n;
        }
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
        if (Rank >= 2)
        {
            off += row * Strides[0];
        }

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
                if (Dims[i] != 1 && Strides[i] != expect)
                {
                    return false;
                }

                expect *= Dims[i];
            }
            return true;
        }
    }

    public static long[] RowMajor(long[] dims)
    {
        var s = new long[dims.Length];
        long e = 1;
        for (int i = dims.Length - 1; i >= 0; i--)
        {
            s[i] = e;
            e *= dims[i];
        }
        return s;
    }
}
