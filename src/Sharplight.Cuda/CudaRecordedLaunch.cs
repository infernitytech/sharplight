namespace Sharplight.Cuda;

/// <summary>One launch of a recorded step: everything <c>cuLaunchKernel</c> needs, resolved for one binding of the symbols.</summary>
/// <param name="Name">Launch name (profiling and diagnostics).</param>
/// <param name="Kernel">The loaded kernel.</param>
/// <param name="GridX">Blocks in the grid, x.</param>
/// <param name="GridY">Blocks in the grid, y.</param>
/// <param name="GridZ">Blocks in the grid, z.</param>
/// <param name="Arguments">The kernel's parameter struct: pointers (device addresses) then 32-bit scalars, passed by value.</param>
internal sealed record CudaRecordedLaunch(string Name, CudaKernel Kernel, uint GridX, uint GridY, uint GridZ, byte[] Arguments);
