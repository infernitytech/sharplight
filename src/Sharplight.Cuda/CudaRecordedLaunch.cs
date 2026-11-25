namespace Sharplight.Cuda;

/// <summary>One launch of a recorded step: everything <c>cuLaunchKernel</c> needs, resolved for one binding of the symbols.</summary>
/// <param name="Name">Launch name (profiling and diagnostics).</param>
/// <param name="ProfileName">Name of the launch kind in profiles (launches of the same kind add up).</param>
/// <param name="Kernel">The loaded kernel.</param>
/// <param name="GridX">Blocks in the grid, x.</param>
/// <param name="GridY">Blocks in the grid, y.</param>
/// <param name="GridZ">Blocks in the grid, z.</param>
/// <param name="Arguments">The kernel's parameter struct: pointers (device addresses) then 32-bit scalars, passed by value.</param>
/// <param name="BarrierBefore">The launch conflicts with a launch since the previous barrier (its hazard batch starts here), so a captured
/// graph runs it after every launch before it; launches of one batch may run side by side.</param>
internal sealed record CudaRecordedLaunch(string Name, string ProfileName, CudaKernel Kernel, uint GridX, uint GridY, uint GridZ, byte[] Arguments,
    bool BarrierBefore = true);
