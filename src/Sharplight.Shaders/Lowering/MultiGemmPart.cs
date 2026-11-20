namespace Sharplight.Shaders.Lowering;

/// <summary>
/// One member's fused split-K launch as the multi-output GEMM needs it (see <see cref="MultiGemmGroup"/>): the activations it reads
/// (the shared f16 or E4M3 copy), its weight, its output, its bias row (applied in its reduction) and the bindings it serves.
/// </summary>
/// <param name="Fp8">True for the fp8 GEMM (gemm_fp8.slang), false for gemm_cm2.</param>
/// <param name="Activations">The activations the split launch reads.</param>
/// <param name="Weight">The member's weight matrix.</param>
/// <param name="Output">The tensor the reduction writes.</param>
/// <param name="Bias">The bias row the reduction adds, or null.</param>
/// <param name="Macros">The member's GEMM macros (format, K, row bytes, strides; N is the member's).</param>
/// <param name="Serves">The bindings the member's split launch serves (its launch condition, more than 8 rows and a K split).</param>
/// <param name="Rows">The member's row count of a binding.</param>
/// <param name="RowsMax">The largest row count of the activations.</param>
/// <param name="Tokens">The member's row-count argument.</param>
internal sealed record MultiGemmPart(bool Fp8, TensorRef Activations, TensorRef Weight, TensorRef Output, TensorRef? Bias,
    Dictionary<string, string> Macros, Func<Bindings, bool> Serves, Func<Bindings, long> Rows, long RowsMax, KernelArg Tokens);
