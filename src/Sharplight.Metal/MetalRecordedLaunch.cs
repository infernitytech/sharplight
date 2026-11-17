using Sharplight.Metal.Interop;

namespace Sharplight.Metal;

/// <summary>One dispatch of a recorded step: everything the encoder needs, resolved for one binding of the symbols.</summary>
/// <param name="Name">Launch name (profiling and diagnostics).</param>
/// <param name="Pipeline">The <c>MTLComputePipelineState</c>.</param>
/// <param name="Threadgroups">Threadgroups in the grid.</param>
/// <param name="Threads">Threads per threadgroup.</param>
/// <param name="Arguments">The kernel's parameter struct: pointers (GPU addresses) then 32-bit scalars, padded to 16 bytes.</param>
/// <param name="BarrierBefore">The launch depends on a launch since the previous barrier.</param>
internal sealed record MetalRecordedLaunch(string Name, nint Pipeline, MTLSize Threadgroups, MTLSize Threads, byte[] Arguments, bool BarrierBefore);
