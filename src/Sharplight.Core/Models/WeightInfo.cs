using Sharplight.Tensors;

namespace Sharplight.Models;

/// <summary>A model parameter: name, logical shape, storage type and where its bytes live.</summary>
public sealed class WeightInfo
{
    /// <summary>Canonical weight name.</summary>
    public required string Name { get; init; }
    /// <summary>Storage type (float or block-quantized).</summary>
    public required DataType Type { get; init; }
    /// <summary>Logical shape outermost-first, e.g. a linear layer is [N_out, K_in].</summary>
    public required Shape Shape { get; init; }
    /// <summary>Where the bytes live.</summary>
    public required IWeightSource Source { get; init; }
    /// <summary>Size of the tensor in bytes, from <see cref="Type"/> and <see cref="Shape"/>.</summary>
    public long ByteSize => Type.ByteSize(Shape.Elements());
    public override string ToString() => $"{Name} {Type}{Shape}";

    /// <summary>
    /// A weight whose bytes are computed on first access (e.g. an importer converting a sign
    /// convention, folding a constant or re-laying out data). The producer must return exactly
    /// <c>type.ByteSize(shape.Elements())</c> bytes.
    /// </summary>
    /// <param name="name">Weight name.</param>
    /// <param name="type">Storage type.</param>
    /// <param name="shape">Logical shape.</param>
    /// <param name="produce">Computes the bytes on first read; the result is cached for the lifetime of the weight.</param>
    public static WeightInfo Derived(string name, DataType type, Shape shape, Func<byte[]> produce)
        => new() { Name = name, Type = type, Shape = shape, Source = new DerivedWeightSource(type.ByteSize(shape.Elements()), produce) };

    /// <summary>Same bytes under a different name and/or shape (element count must match).</summary>
    /// <param name="name">The new name.</param>
    /// <param name="shape">The new shape, or <see langword="null"/> to keep the current one.</param>
    /// <exception cref="ArgumentException">The element counts differ.</exception>
    public WeightInfo As(string name, Shape? shape = null)
    {
        var s = shape ?? Shape;
        if (s.Elements() != Shape.Elements())
        {
            throw new ArgumentException($"{Name}: cannot view {Shape} as {s}.");
        }

        return new() { Name = name, Type = Type, Shape = s, Source = Source };
    }

    /// <summary>
    /// Zero-copy view of rows [firstRow, firstRow + rows) of a tensor viewed as [Shape[0], rest...] (e.g. splitting a fused
    /// q/k/v or gate/up projection). Rows are contiguous in every storage type, including block-quantized ones.
    /// </summary>
    /// <param name="name">Name of the slice.</param>
    /// <param name="firstRow">First row of the slice.</param>
    /// <param name="rows">Number of rows in the slice.</param>
    /// <exception cref="ArgumentOutOfRangeException">The rows fall outside the tensor.</exception>
    public WeightInfo RowSlice(string name, long firstRow, long rows)
    {
        long total = Shape[0].Fixed;
        if (firstRow < 0 || rows <= 0 || firstRow + rows > total)
        {
            throw new ArgumentOutOfRangeException(nameof(rows), $"{Name}: rows [{firstRow}, {firstRow + rows}) outside [0, {total}).");
        }

        long rowElems = Shape.Elements() / total;
        long rowBytes = Type.ByteSize(rowElems);
        var dims = new List<Dim> { rows };
        for (int d = 1; d < Shape.Rank; d++)
        {
            dims.Add(Shape[d]);
        }

        return new() { Name = name, Type = Type, Shape = new Shape(dims), Source = new SliceWeightSource(Source, firstRow * rowBytes, rows * rowBytes) };
    }

    /// <summary>
    /// Stacks equally shaped tensors along a new outermost axis (per-expert matrices into one [E, N, K] tensor). The result reads
    /// its parts lazily; nothing is copied into memory up front.
    /// </summary>
    /// <param name="name">Name of the stacked weight.</param>
    /// <param name="parts">Tensors to stack; all must have the same type and shape.</param>
    /// <exception cref="ArgumentException">The parts differ in type or shape.</exception>
    public static WeightInfo Stack(string name, IReadOnlyList<WeightInfo> parts)
    {
        var first = parts[0];
        foreach (var w in parts)
        {
            if (w.Type != first.Type || !w.Shape.Equals(first.Shape))
            {
                throw new ArgumentException($"{name}: cannot stack {w} with {first}.");
            }
        }

        var dims = new List<Dim> { parts.Count };
        for (int d = 0; d < first.Shape.Rank; d++)
        {
            dims.Add(first.Shape[d]);
        }

        return new() { Name = name, Type = first.Type, Shape = new Shape(dims), Source = new StackedWeightSource([.. parts.Select(w => w.Source)]) };
    }

    /// <summary>Reads the whole tensor.</summary>
    public byte[] ReadAll()
    {
        var b = new byte[Source.ByteLength];
        Source.Read(0, b);
        return b;
    }
}
