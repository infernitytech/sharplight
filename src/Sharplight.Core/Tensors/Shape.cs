using System.Collections.Immutable;
using System.Text;

namespace Sharplight.Tensors;

/// <summary>
/// Logical tensor shape, outermost dimension first (row-major, like NumPy/ONNX).
/// Only the outermost dimension may be symbolic; this keeps all strides static so kernels can be
/// specialized at compile time while the batch/token count stays dynamic.
/// </summary>
public sealed class Shape : IEquatable<Shape>
{
    /// <summary>The dimensions, outermost first.</summary>
    public ImmutableArray<Dim> Dims { get; }
    /// <summary>Number of dimensions (0 for a scalar).</summary>
    public int Rank => Dims.Length;

    /// <summary>
    /// Creates a shape.
    /// </summary>
    /// <param name="dims">Dimensions, outermost first.</param>
    /// <exception cref="ArgumentException">A dimension other than the outermost is symbolic.</exception>
    public Shape(params ReadOnlySpan<Dim> dims)
    {
        Dims = [.. dims];
        for (int i = 1; i < Dims.Length; i++)
        {
            if (Dims[i].IsSymbolic)
            {
                throw new ArgumentException($"Only the outermost dimension may be symbolic (shape {this}).");
            }
        }
    }

    /// <summary>
    /// Creates a shape from a sequence of dimensions.
    /// </summary>
    /// <param name="dims">Dimensions, outermost first.</param>
    /// <exception cref="ArgumentException">A dimension other than the outermost is symbolic.</exception>
    public Shape(IEnumerable<Dim> dims) : this([.. dims]) { }

    /// <summary>The rank-0 shape (one element).</summary>
    public static Shape Scalar { get; } = new();

    /// <summary>
    /// Gets a dimension.
    /// </summary>
    /// <param name="axis">Axis index; negative values count from the end.</param>
    public Dim this[int axis] => Dims[axis < 0 ? Rank + axis : axis];

    /// <summary>True when no dimension is symbolic.</summary>
    public bool IsStatic => Rank == 0 || !Dims[0].IsSymbolic;
    /// <summary>Symbol of the outermost dimension, or <see langword="null"/> when the shape is static.</summary>
    public string? Symbol => Rank > 0 ? Dims[0].Symbol : null;

    /// <summary>Product of all dimensions except the (possibly symbolic) outermost one.</summary>
    public long InnerElements
    {
        get
        {
            long n = 1;
            for (int i = 1; i < Rank; i++)
            {
                n *= Dims[i].Fixed;
            }

            return n;
        }
    }

    /// <summary>
    /// Number of elements.
    /// </summary>
    /// <param name="bindings">Values of the symbol of the outermost dimension; required when the shape is symbolic.</param>
    /// <exception cref="InvalidOperationException">The shape is symbolic and the symbol is not bound.</exception>
    public long Elements(IReadOnlyDictionary<string, long>? bindings = null)
    {
        if (Rank == 0)
        {
            return 1;
        }

        return Dims[0].Resolve(bindings) * InnerElements;
    }

    /// <summary>
    /// Gets the concrete extent of every dimension.
    /// </summary>
    /// <param name="bindings">Values of the symbol of the outermost dimension.</param>
    /// <exception cref="InvalidOperationException">The shape is symbolic and the symbol is not bound.</exception>
    public long[] Resolve(IReadOnlyDictionary<string, long>? bindings)
    {
        var r = new long[Rank];
        for (int i = 0; i < Rank; i++)
        {
            r[i] = Dims[i].Resolve(bindings);
        }

        return r;
    }

    /// <summary>
    /// Returns a copy with one dimension replaced.
    /// </summary>
    /// <param name="axis">Axis to replace; negative values count from the end.</param>
    /// <param name="d">The new dimension.</param>
    public Shape WithDim(int axis, Dim d)
    {
        if (axis < 0)
        {
            axis += Rank;
        }

        var b = Dims.ToBuilder();
        b[axis] = d;
        return new Shape(b.ToImmutable());
    }

    /// <summary>
    /// Returns a copy with the last dimension replaced.
    /// </summary>
    /// <param name="d">The new dimension.</param>
    public Shape WithLast(Dim d) => WithDim(Rank - 1, d);

    /// <summary>Whether both shapes have the same dimensions.</summary>
    public bool Equals(Shape? other) => other is not null && Dims.SequenceEqual(other.Dims);
    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as Shape);
    /// <inheritdoc/>
    public override int GetHashCode()
    {
        var h = new HashCode();
        foreach (var d in Dims)
        {
            h.Add(d);
        }

        return h.ToHashCode();
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < Rank; i++)
        {
            if (i > 0)
            {
                sb.Append(", ");
            }

            sb.Append(Dims[i]);
        }
        return sb.Append(']').ToString();
    }
}
