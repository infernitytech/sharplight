using System.Collections.Immutable;
using System.Text;

namespace Sharplight.Tensors;

/// <summary>
/// A tensor dimension: either a fixed extent or a named symbol (e.g. <c>T</c> = number of tokens in
/// the current batch) that is bound when a graph is executed.
/// </summary>
public readonly record struct Dim
{
    public long Fixed { get; }
    public string? Symbol { get; }
    /// <summary>Upper bound for symbolic dims (used for memory planning). 0 = unknown.</summary>
    public bool IsSymbolic => Symbol is not null;

    Dim(long value, string? symbol) { Fixed = value; Symbol = symbol; }

    public static Dim Of(long value)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(value);
        return new Dim(value, null);
    }

    public static Dim Sym(string symbol) => new(0, symbol);

    public static implicit operator Dim(long value) => Of(value);
    public static implicit operator Dim(int value) => Of(value);

    public long Resolve(IReadOnlyDictionary<string, long>? bindings)
    {
        if (Symbol is null) return Fixed;
        if (bindings is not null && bindings.TryGetValue(Symbol, out var v)) return v;
        throw new InvalidOperationException($"Symbolic dimension '{Symbol}' is not bound.");
    }

    public override string ToString() => Symbol ?? Fixed.ToString();
}

/// <summary>
/// Logical tensor shape, outermost dimension first (row-major, like NumPy/ONNX).
/// Only the outermost dimension may be symbolic; this keeps all strides static so kernels can be
/// specialized at compile time while the batch/token count stays dynamic.
/// </summary>
public sealed class Shape : IEquatable<Shape>
{
    public ImmutableArray<Dim> Dims { get; }
    public int Rank => Dims.Length;

    public Shape(params ReadOnlySpan<Dim> dims)
    {
        Dims = [.. dims];
        for (int i = 1; i < Dims.Length; i++)
            if (Dims[i].IsSymbolic)
                throw new ArgumentException($"Only the outermost dimension may be symbolic (shape {this}).");
    }

    public Shape(IEnumerable<Dim> dims) : this([.. dims]) { }

    public static Shape Scalar { get; } = new();

    public Dim this[int axis] => Dims[axis < 0 ? Rank + axis : axis];

    public bool IsStatic => Rank == 0 || !Dims[0].IsSymbolic;
    public string? Symbol => Rank > 0 ? Dims[0].Symbol : null;

    /// <summary>Product of all dimensions except the (possibly symbolic) outermost one.</summary>
    public long InnerElements
    {
        get
        {
            long n = 1;
            for (int i = 1; i < Rank; i++) n *= Dims[i].Fixed;
            return n;
        }
    }

    public long Elements(IReadOnlyDictionary<string, long>? bindings = null)
    {
        if (Rank == 0) return 1;
        return Dims[0].Resolve(bindings) * InnerElements;
    }

    public long[] Resolve(IReadOnlyDictionary<string, long>? bindings)
    {
        var r = new long[Rank];
        for (int i = 0; i < Rank; i++) r[i] = Dims[i].Resolve(bindings);
        return r;
    }

    public Shape WithDim(int axis, Dim d)
    {
        if (axis < 0) axis += Rank;
        var b = Dims.ToBuilder();
        b[axis] = d;
        return new Shape(b.ToImmutable());
    }

    public Shape WithLast(Dim d) => WithDim(Rank - 1, d);

    public bool Equals(Shape? other) => other is not null && Dims.SequenceEqual(other.Dims);
    public override bool Equals(object? obj) => Equals(obj as Shape);
    public override int GetHashCode()
    {
        var h = new HashCode();
        foreach (var d in Dims) h.Add(d);
        return h.ToHashCode();
    }

    public override string ToString()
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < Rank; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(Dims[i]);
        }
        return sb.Append(']').ToString();
    }
}
