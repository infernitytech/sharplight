namespace Sharplight.Shaders.Lowering;

/// <summary>One kernel choice for a launch (e.g. matvec vs gemm, selected by token count).</summary>
public sealed record KernelVariant(Func<Bindings, bool> When, CompiledKernel Kernel, Func<Bindings, Grid> Grid);
