namespace Sharplight.Shaders.Lowering;

/// <summary>One kernel choice for a launch (e.g. matvec vs gemm, selected by token count).</summary>
/// <param name="When">Condition on the bindings under which this variant runs.</param>
/// <param name="Kernel">The compiled kernel.</param>
/// <param name="Grid">Workgroup count for the bindings.</param>
public sealed record KernelVariant(Func<Bindings, bool> When, CompiledKernel Kernel, Func<Bindings, Grid> Grid);
