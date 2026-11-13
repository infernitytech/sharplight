namespace Sharplight.Shaders;

/// <summary>Process-wide kernel compilation totals, see <see cref="SlangCompiler.Statistics"/>.</summary>
/// <param name="Compiled">Kernels compiled with Slang.</param>
/// <param name="DiskHits">Kernels read from the on-disk kernel cache.</param>
/// <param name="CompileTime">Summed wall time of the Slang compilations (exceeds elapsed time when compiles run in parallel).</param>
public sealed record CompileStatistics(int Compiled, int DiskHits, TimeSpan CompileTime);
