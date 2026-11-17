using Sharplight.Metal.Interop;

namespace Sharplight.Metal;

/// <summary>A compute pipeline state and the threadgroup size its kernel was written for.</summary>
/// <param name="Handle">The retained <c>MTLComputePipelineState</c>.</param>
/// <param name="Threads">Threads per threadgroup (the kernel's <c>[numthreads]</c>).</param>
internal sealed record MetalPipeline(nint Handle, MTLSize Threads);
