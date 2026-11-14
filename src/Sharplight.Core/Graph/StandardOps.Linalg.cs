using Sharplight.Tensors;

namespace Sharplight.Graph;

// Dense linear algebra beyond matrix products.
public static partial class StandardOps
{
    /// <summary>
    /// Determinant of square matrices: x [..., M, M] → f32 [...] (rank 0 for a single matrix). Computed by LU decomposition with
    /// partial pivoting.
    /// </summary>
    public static readonly OpSchema Det = new("Det", OpCategory.Reduction, 1, 1, (i, a) =>
    {
        var x = i[0].Shape;
        if (x.Rank < 2 || x[-1] != x[-2])
        {
            throw new ArgumentException($"Det expects square matrices [..., M, M], got {x}.");
        }

        return [new(DataType.F32, new Shape(x.Dims.Take(x.Rank - 2)))];
    });
}
