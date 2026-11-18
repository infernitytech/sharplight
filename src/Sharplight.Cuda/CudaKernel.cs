namespace Sharplight.Cuda;

/// <summary>A loaded kernel: its module, the entry function and the block size it was written for.</summary>
/// <param name="Module">The <c>CUmodule</c> (owned by the device).</param>
/// <param name="Function">The <c>CUfunction</c> of the entry point.</param>
/// <param name="ThreadsX">Block size, x.</param>
/// <param name="ThreadsY">Block size, y.</param>
/// <param name="ThreadsZ">Block size, z.</param>
internal sealed record CudaKernel(nint Module, nint Function, uint ThreadsX, uint ThreadsY, uint ThreadsZ);
