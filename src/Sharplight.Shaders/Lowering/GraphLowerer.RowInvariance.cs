using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// Row counts up to which every row of a step must come out bit-identical to a step of that row alone
    /// (<see cref="Runtime.CompileOptions.RowInvariantRows"/>, <c>SHARPLIGHT_ROW_INVARIANT</c>), capped at the matvec range.
    /// Such steps keep the single-row arithmetic: the f32 matvec (or, for formats whose single rows take the integer-dot matvec,
    /// its single-row layout) with the same K split per row, and decode attention with the single-row key split. The batched
    /// kernels they replace (integer-dot matvec on q8 activations, skinny tensor-core GEMM on f16 activations, fewer key
    /// splits) round differently, so speculative verification rows would not reproduce the tokens of plain decoding.
    /// </summary>
    private long InvariantRows => Math.Clamp(DiagnosticSwitches.RowInvariantRows ?? _opt.Compile.RowInvariantRows, 0, MatvecMaxRows);

    /// <summary>
    /// First row count the integer-dot matvec serves for weights of <paramref name="type"/>: 1 where single rows use it too,
    /// else above both the f32 single-row matvec and the row-invariant range.
    /// </summary>
    private long IntDotFirstRow(DataType type)
    {
        long min = IntDotMinRows(type);
        return min > 1 ? Math.Max(min, InvariantRows + 1) : min;
    }

    /// <summary>True when a matvec bucket serving up to <paramref name="columns"/> rows (above half of it) holds row-invariant rows.</summary>
    private bool InvariantBucket(long columns) => columns > 1 && columns / 2 < InvariantRows;
}
