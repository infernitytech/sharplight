using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>Strided view of a tensor inside a buffer. Strides and shape are in elements; the outermost
/// dimension may be symbolic (<see cref="Symbol"/>), in which case Shape[0] holds its upper bound.</summary>
/// <param name="Buffer">The buffer holding the data.</param>
/// <param name="ByteOffset">Offset of the first element from the buffer start.</param>
/// <param name="Type">Element type.</param>
/// <param name="Shape">Extent per axis, in elements (the upper bound for a symbolic outermost axis).</param>
/// <param name="Strides">Distance between neighbouring elements per axis, in elements.</param>
/// <param name="Symbol">Name of the symbolic outermost dimension, or null when the shape is static.</param>
public sealed record TensorRef(BufferDecl Buffer, long ByteOffset, DataType Type, long[] Shape, long[] Strides, string? Symbol)
{
    /// <summary>Number of axes.</summary>
    public int Rank => Shape.Length;

    /// <summary>True when the elements are contiguous in row-major order (size-1 axes may have any stride).</summary>
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

    /// <summary>Element count at the shape's upper bounds.</summary>
    public long MaxElements => Shape.Aggregate(1L, (a, b) => a * b);
    /// <summary>Bytes covered at the shape's upper bounds (the span of a strided view, or the packed size of block formats).</summary>
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

    /// <summary>Row-major strides of a dense tensor of <paramref name="shape"/>.</summary>
    public static long[] DenseStrides(long[] shape)
    {
        var s = new long[shape.Length];
        long acc = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            s[i] = acc;
            acc *= shape[i];
        }
        return s;
    }
}
