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
    /// <summary>
    /// The order of the stored columns of a matrix weight, or null for the natural order: stored column j multiplies input element
    /// <c>InputOrder[j]</c>, so a product reads its input gathered along the last axis by this permutation first
    /// (<see cref="Graph.GraphBuilder.MatMul"/> does). Set for act-order (GPTQ <c>desc_act</c>) checkpoints, whose quantization
    /// groups are contiguous only in the permuted order. Kept by renames and row slices; stacked or concatenated parts must agree.
    /// </summary>
    public int[]? InputOrder { get; init; }
    /// <summary>
    /// For a block-scaled matrix weight (one scale per row and block of columns, e.g. <see cref="DataType.F8E4M3_B128"/>): how many
    /// consecutive rows of each matrix (the second-to-last axis, counted from its row 0) are known to share every block's scale, as
    /// checkpoints with 128 x 128 scale blocks have them; 0 when unknown. Kernels may then read one scale per row group. Kept by the
    /// views below only where the groups stay aligned; stored in packages.
    /// </summary>
    public int ScaleRowBlock { get; init; }
    /// <summary>
    /// What the weight's quantization format defines for the activations of its products (weight-only, or activations quantized
    /// too); <see cref="ActivationScheme.Unspecified"/> when the source does not say. Importers set it from the checkpoint's
    /// quantization config; kept by every view, stored in packages.
    /// </summary>
    public ActivationScheme Activations { get; init; }
    /// <summary>Size of the tensor in bytes, from <see cref="Type"/> and <see cref="Shape"/>.</summary>
    public long ByteSize => Type.ByteSize(Shape.Elements());
    /// <inheritdoc/>
    public override string ToString() => $"{Name} {Type}{Shape}";

    /// <summary>
    /// A weight whose bytes are computed on first access (e.g. an importer converting a sign
    /// convention, folding a constant or re-laying out data). The producer must return exactly
    /// <c>type.ByteSize(shape.Elements())</c> bytes.
    /// </summary>
    /// <param name="name">Weight name.</param>
    /// <param name="type">Storage type.</param>
    /// <param name="shape">Logical shape.</param>
    /// <param name="produce">Computes the bytes on first read; the result is cached until <see cref="IWeightSource.Release"/> (a package writer releases it after writing the weight).</param>
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

        if (InputOrder is not null && !s[-1].Equals(Shape[-1]))
        {
            throw new ArgumentException($"{Name}: a weight with an input order cannot change its last dimension ({Shape} as {s}).");
        }

        return new()
        {
            Name = name, Type = Type, Shape = s, Source = Source, InputOrder = InputOrder, ScaleRowBlock = ViewScaleRowBlock(s), Activations = Activations,
        };
    }

    /// <summary>This weight (same name, type, shape, bytes and input order) with the given <see cref="ScaleRowBlock"/> and <see cref="Activations"/>.</summary>
    internal WeightInfo WithFacts(int scaleRowBlock, ActivationScheme activations)
        => new() { Name = Name, Type = Type, Shape = Shape, Source = Source, InputOrder = InputOrder, ScaleRowBlock = scaleRowBlock, Activations = activations };

    /// <summary><see cref="ScaleRowBlock"/> of a view of this tensor as <paramref name="shape"/> (the same elements, row-major).</summary>
    private int ViewScaleRowBlock(Shape shape)
    {
        int block = ScaleRowBlock;
        if (block <= 0 || shape.Rank < 2 || Shape.Rank < 2 || !shape[-1].Equals(Shape[-1]) || shape[-2].IsSymbolic || Shape[-2].IsSymbolic)
        {
            return 0;
        }

        return Shape[-2].Fixed % block == 0 && shape[-2].Fixed % block == 0 ? block : 0;
    }

    /// <summary>The <see cref="ScaleRowBlock"/> shared by all <paramref name="parts"/> (0 when they differ or <paramref name="aligned"/> fails for one).</summary>
    private static int CommonScaleRowBlock(IReadOnlyList<WeightInfo> parts, Func<WeightInfo, bool> aligned)
    {
        int block = parts[0].ScaleRowBlock;
        return block > 0 && parts.All(w => w.ScaleRowBlock == block && aligned(w)) ? block : 0;
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

        return new()
        {
            Name = name, Type = Type, Shape = new Shape(dims), Source = new SliceWeightSource(Source, firstRow * rowBytes, rows * rowBytes),
            InputOrder = InputOrder,
            // Slices of a stack keep whole matrices; slices of a matrix keep the groups where they start on one.
            ScaleRowBlock = Shape.Rank > 2 || (ScaleRowBlock > 0 && firstRow % ScaleRowBlock == 0) ? ScaleRowBlock : 0,
            Activations = Activations,
        };
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

        return new()
        {
            Name = name, Type = first.Type, Shape = new Shape(dims), Source = new StackedWeightSource([.. parts.Select(w => w.Source)]),
            InputOrder = CommonInputOrder(name, parts),
            ScaleRowBlock = first.Shape.Rank >= 2 ? CommonScaleRowBlock(parts, _ => true) : 0,
            Activations = CommonActivations(parts),
        };
    }

    /// <summary>
    /// Concatenates tensors along their outermost axis (e.g. row blocks gathered from an interleaved fused projection). The parts
    /// must agree in type and in every dimension but the first; the result reads them lazily, nothing is copied up front.
    /// </summary>
    /// <param name="name">Name of the result.</param>
    /// <param name="parts">Tensors to concatenate, in order.</param>
    /// <exception cref="ArgumentException">The parts differ in type or row shape.</exception>
    public static WeightInfo ConcatRows(string name, IReadOnlyList<WeightInfo> parts)
    {
        var first = parts[0];
        long rows = 0;
        foreach (var w in parts)
        {
            bool sameRows = w.Shape.Rank == first.Shape.Rank && Enumerable.Range(1, first.Shape.Rank - 1).All(d => w.Shape[d].Equals(first.Shape[d]));
            if (w.Type != first.Type || !sameRows)
            {
                throw new ArgumentException($"{name}: cannot concatenate {w} with {first}.");
            }

            rows += w.Shape[0].Fixed;
        }

        var dims = new List<Dim> { rows };
        for (int d = 1; d < first.Shape.Rank; d++)
        {
            dims.Add(first.Shape[d]);
        }

        return new()
        {
            Name = name, Type = first.Type, Shape = new Shape(dims), Source = new ConcatenatedWeightSource([.. parts.Select(w => w.Source)]),
            InputOrder = CommonInputOrder(name, parts),
            // Matrices joined by rows keep the groups where every part holds whole groups; stacks joined by matrices keep them.
            ScaleRowBlock = CommonScaleRowBlock(parts, w => w.Shape.Rank > 2 || w.Shape[0].Fixed % w.ScaleRowBlock == 0),
            Activations = CommonActivations(parts),
        };
    }

    /// <summary>
    /// The <see cref="Activations"/> of a tensor joined from <paramref name="parts"/>: weight-only when one part is (its products must
    /// keep full-precision activations), quantized when all are, else unspecified.
    /// </summary>
    private static ActivationScheme CommonActivations(IReadOnlyList<WeightInfo> parts)
    {
        if (parts.Any(w => w.Activations == ActivationScheme.Unquantized))
        {
            return ActivationScheme.Unquantized;
        }

        return parts.All(w => w.Activations == ActivationScheme.Quantized) ? ActivationScheme.Quantized : ActivationScheme.Unspecified;
    }

    /// <summary>The input order shared by all <paramref name="parts"/>.</summary>
    /// <exception cref="ArgumentException">The parts have different input orders.</exception>
    private static int[]? CommonInputOrder(string name, IReadOnlyList<WeightInfo> parts)
    {
        var order = parts[0].InputOrder;
        foreach (var w in parts)
        {
            if (!ReferenceEquals(w.InputOrder, order) && (w.InputOrder is null || order is null || !w.InputOrder.AsSpan().SequenceEqual(order)))
            {
                throw new ArgumentException($"{name}: cannot combine weights with different input orders ({w.Name}, {parts[0].Name}).");
            }
        }

        return order;
    }

    /// <summary>Gathers row blocks (first row, row count) of this tensor, in order, into one lazily read tensor.</summary>
    /// <param name="name">Name of the result.</param>
    /// <param name="blocks">The row ranges to take.</param>
    public WeightInfo GatherRows(string name, IEnumerable<(long FirstRow, long Rows)> blocks)
        => ConcatRows(name, [.. blocks.Select(block => RowSlice(name, block.FirstRow, block.Rows))]);

    /// <summary>
    /// Views each outer row (of <paramref name="a"/>·<paramref name="b"/> elements, a-major) as an a×b matrix and transposes it, lazily:
    /// the result is [rows, b·a], b-major. E.g. a convolution kernel [out, C, kh, kw] with a = C, b = kh·kw becomes the linear map of
    /// window rows laid out (kh, kw, C).
    /// </summary>
    /// <param name="name">Name of the result.</param>
    /// <param name="a">Outer extent of each row's matrix.</param>
    /// <param name="b">Inner extent of each row's matrix.</param>
    /// <exception cref="ArgumentException">The tensor is quantized or its rows do not hold a·b elements.</exception>
    public WeightInfo TransposeRowBlocks(string name, long a, long b)
    {
        long rows = Shape[0].Fixed;
        if (Type.IsQuantized || InputOrder is not null || Shape.Elements() != rows * a * b)
        {
            throw new ArgumentException($"{Name}: cannot view rows of {Type}{Shape} as {a}×{b} matrices.");
        }

        var source = this;
        int eb = Type.ElementSize;
        return Derived(name, Type, new Shape(rows, a * b), () =>
        {
            var src = source.ReadAll();
            var dst = new byte[src.Length];
            for (long r = 0; r < rows; r++)
            {
                long row = r * a * b;
                for (long i = 0; i < a; i++)
                {
                    for (long j = 0; j < b; j++)
                    {
                        Buffer.BlockCopy(src, (int)((row + i * b + j) * eb), dst, (int)((row + j * a + i) * eb), eb);
                    }
                }
            }

            return dst;
        });
    }

    /// <summary>Reads the whole tensor.</summary>
    public byte[] ReadAll()
    {
        var b = new byte[Source.ByteLength];
        Source.Read(0, b);
        return b;
    }
}
