using System.Text;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>Source-text helpers shared by the generated elementwise and row-normalization kernels.</summary>
internal static class ElementwiseCodegen
{
    /// <summary>Elements spanned by one step of each axis of a dense tensor of the given shape (row-major pitches).</summary>
    public static long[] Pitches(long[] shape)
    {
        var pitches = new long[shape.Length];
        long running = 1;
        for (int axis = shape.Length - 1; axis >= 0; axis--)
        {
            pitches[axis] = running;
            running *= shape[axis];
        }

        return pitches;
    }

    /// <summary>Product of all dimensions.</summary>
    public static long ElementCount(long[] shape) => shape.Aggregate(1L, (a, b) => a * b);

    /// <summary>Slang declarations <c>d0 .. dN</c> splitting the flat index <c>idx</c> into coordinates of <paramref name="shape"/>.</summary>
    public static void AppendCoordinates(StringBuilder sb, string indent, long[] shape, long[] pitches)
    {
        for (int axis = 0; axis < shape.Length; axis++)
        {
            sb.AppendLine(axis == 0
                ? $"{indent}uint d0 = idx / {pitches[0]}u;"
                : $"{indent}uint d{axis} = (idx / {pitches[axis]}u) % {shape[axis]}u;");
        }
    }

    /// <summary>
    /// Slang expression for the element offset of <paramref name="r"/> given coordinates <c>d0 .. dN</c> of the broadcast
    /// <paramref name="shape"/> (size-1 axes of <paramref name="r"/> broadcast; leading axes may be missing).
    /// </summary>
    public static string Offset(TensorRef r, long[] shape)
    {
        int rank = shape.Length;
        var terms = new List<string>();
        for (int axis = 0; axis < rank; axis++)
        {
            int refAxis = r.Rank - rank + axis;
            if (refAxis < 0)
            {
                continue;
            }

            long stride = r.Shape[refAxis] == 1 && shape[axis] != 1 ? 0 : r.Strides[refAxis];
            if (stride == 0 || shape[axis] == 1)
            {
                continue;
            }

            terms.Add(stride == 1 ? $"d{axis}" : $"d{axis} * {stride}u");
        }

        return terms.Count == 0 ? "0" : string.Join(" + ", terms);
    }

    /// <summary>Slang expression loading element <c>in{index}</c> of <paramref name="r"/> as f32.</summary>
    public static string Load(TensorRef r, int index, long[] shape)
    {
        string element = $"p.in{index}[{Offset(r, shape)}]";
        if (r.Type == DataType.F32)
        {
            return element;
        }

        if (r.Type == DataType.F16)
        {
            return $"h2f({element})";
        }

        if (r.Type == DataType.BF16)
        {
            return $"bf2f({element})";
        }

        if (r.Type == DataType.I32 || r.Type == DataType.I64 || r.Type == DataType.U32)
        {
            return $"float({element})";
        }

        throw new NotSupportedException($"elementwise input type {r.Type}");
    }

    /// <summary>Slang expression converting a computed value to the stored element of type <paramref name="t"/> (16-bit floats as raw bits).</summary>
    public static string Store(DataType t, string expr)
    {
        if (t == DataType.F16)
        {
            return $"f2h({expr})";
        }

        return t == DataType.BF16 ? $"f2bf({expr})" : expr;
    }

    /// <summary>Slang pointee type used to read tensors of <paramref name="t"/> (16-bit floats are read as raw bits).</summary>
    public static string PointerType(DataType t)
    {
        if (t == DataType.F32)
        {
            return "float";
        }

        if (t == DataType.I32)
        {
            return "int";
        }

        if (t == DataType.I64)
        {
            return "int64_t";
        }

        return t == DataType.U32 ? "uint" : "uint16_t";
    }
}
