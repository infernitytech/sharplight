namespace Sharplight.Shaders.Lowering;

/// <summary>
/// One member's launch on the hand-written f16 (or FP8) GEMM's prompt-sized tiles as the multi-output form needs it (see
/// <see cref="MultiGemmGroup.TallParts"/>): the f16 activations it reads, its weight, its output, its bias row (added in its store)
/// and the rows it serves.
/// </summary>
/// <param name="Activations">The f16 copy of the activations the launch reads.</param>
/// <param name="Weight">The member's f16 weight matrix.</param>
/// <param name="Output">The tensor the launch writes (dense rows of the member's width).</param>
/// <param name="Bias">The bias row the store adds, or null.</param>
/// <param name="RangeScales">The byte offset of the activations' row-range scale words, or null when the copy is not row-ranged.</param>
/// <param name="Rows">The member's row count of a binding.</param>
/// <param name="MinRows">The fewest rows the launch serves.</param>
/// <param name="MaxRows">The largest row count of the activations.</param>
/// <param name="Serves">The bindings the member's launch serves (its launch condition and at least <paramref name="MinRows"/> rows).</param>
/// <param name="Tokens">The member's row-count argument.</param>
/// <param name="Fp8">True for a member on the hand-written FP8 GEMM's 128-row tiles (<c>gemm_fp8_tma.cu</c>): E4M3 activation rows and
/// a grouped E4M3 weight instead of f16 ones.</param>
/// <param name="UniformScales">FP8: the weight's 128-row groups share each block's scale.</param>
internal sealed record MultiGemmTallPart(TensorRef Activations, TensorRef Weight, TensorRef Output, TensorRef? Bias, long? RangeScales,
    Func<Bindings, long> Rows, long MinRows, long MaxRows, Func<Bindings, bool> Serves, KernelArg Tokens, bool Fp8 = false, bool UniformScales = false);
