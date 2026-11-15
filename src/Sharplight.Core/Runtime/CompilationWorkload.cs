namespace Sharplight.Runtime;

/// <summary>
/// Compiles everything a model needs on <paramref name="device"/>, exactly as its loaders would (the same graphs, bounds and
/// placement), with <see cref="CompileOptions.CacheFor"/> set to <see cref="CompilationArtifacts.For"/> of
/// <paramref name="artifacts"/> so every kernel is recorded. The portable stage
/// (<see cref="IExecutionProvider.CompilePortableAsync"/>) runs it on a compile-only device of the target profile.
/// </summary>
/// <param name="device">The device to compile on (compile-only: the segments it returns cannot run).</param>
/// <param name="artifacts">Records the generated code.</param>
/// <param name="cancellationToken">Cancels the compilation between kernels.</param>
public delegate void CompilationWorkload(IComputeDevice device, CompilationArtifacts artifacts, CancellationToken cancellationToken);
