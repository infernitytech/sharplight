using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

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
