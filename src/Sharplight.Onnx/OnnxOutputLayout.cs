namespace Sharplight.Onnx;

/// <summary>
/// How a graph output's IR tensor maps to the ONNX tensor when the two differ. Only the outermost IR dimension may be symbolic, so an ONNX
/// output whose data-dependent extent sits on an inner axis is stored with that extent moved to the front: StringNormalizer of
/// <c>[1, C]</c> yields IR <c>[count]</c> behind a unit dim, NonZero yields IR <c>[count, rank]</c> for ONNX <c>[rank, count]</c>, Unique
/// along an inner axis yields its slices along the outer IR axis. The ONNX tensor is <see cref="LeadingUnits"/> unit dims prepended to
/// the IR tensor, then transposed by <see cref="Permutation"/>. Listed under <see cref="OnnxGraphAdapter.OutputLayoutsProperty"/>.
/// </summary>
/// <param name="LeadingUnits">Unit dims in front of the IR shape.</param>
/// <param name="Permutation">ONNX axis <c>i</c> is axis <c>Permutation[i]</c> of the IR shape with the unit dims prepended; empty: no transpose.</param>
public sealed record OnnxOutputLayout(int LeadingUnits, int[] Permutation)
{
    /// <summary>The ONNX shape of an output whose IR tensor has shape <paramref name="irShape"/>.</summary>
    /// <param name="irShape">The resolved IR shape (as reported by the compiled model).</param>
    /// <returns>The ONNX shape.</returns>
    public long[] OnnxShape(ReadOnlySpan<long> irShape)
    {
        long[] expanded = Expanded(irShape);
        return Permutation.Length == 0 ? expanded : [.. Permutation.Select(p => expanded[p])];
    }

    /// <summary>Reorders the row-major elements of an IR output into the row-major order of the ONNX tensor.</summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="data">The IR output's elements.</param>
    /// <param name="irShape">The resolved IR shape.</param>
    /// <returns>The ONNX tensor's elements (a copy).</returns>
    public T[] ToOnnx<T>(ReadOnlySpan<T> data, ReadOnlySpan<long> irShape)
    {
        long[] expanded = Expanded(irShape);
        if (Permutation.Length == 0)
        {
            return data.ToArray();
        }

        int rank = expanded.Length;
        var sourceStrides = new long[rank];
        long stride = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            sourceStrides[d] = stride;
            stride *= expanded[d];
        }

        long[] shape = OnnxShape(irShape);
        var result = new T[data.Length];
        var index = new long[rank];
        for (long i = 0; i < result.Length; i++)
        {
            long source = 0;
            for (int d = 0; d < rank; d++)
            {
                source += index[d] * sourceStrides[Permutation[d]];
            }

            result[i] = data[(int)source];
            for (int d = rank - 1; d >= 0; d--)
            {
                if (++index[d] < shape[d])
                {
                    break;
                }

                index[d] = 0;
            }
        }

        return result;
    }

    private long[] Expanded(ReadOnlySpan<long> irShape) => [.. Enumerable.Repeat(1L, LeadingUnits), .. irShape];
}
