namespace Sharplight.Shaders.Lowering;

/// <summary>
/// One member's single-row split-K matvec launch as the multi-output matvec needs it (see <see cref="MultiGemmGroup.MatvecParts"/>):
/// the f32 activations it reads, its weight, its output, the macros of its matvec variants and the bindings its single-row launch serves.
/// </summary>
/// <param name="Activations">The f32 activation rows the launch reads.</param>
/// <param name="Weight">The member's weight matrix.</param>
/// <param name="Output">The tensor the launch writes.</param>
/// <param name="Macros">The macros its matvec variants are compiled from (format, K, strides, row bytes).</param>
/// <param name="Serves">The bindings its single-row variant serves: its launch condition and one row.</param>
/// <param name="Tokens">The member's row-count argument.</param>
internal sealed record MultiMatvecPart(TensorRef Activations, TensorRef Weight, TensorRef Output, Dictionary<string, string> Macros,
    Func<Bindings, bool> Serves, KernelArg Tokens);
