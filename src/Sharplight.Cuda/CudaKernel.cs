using Sharplight.Runtime;

namespace Sharplight.Cuda;

/// <summary>A loaded kernel: its module, the entry function and the block size it was written for.</summary>
/// <param name="Module">The <c>CUmodule</c> (owned by the device).</param>
/// <param name="Function">The <c>CUfunction</c> of the entry point.</param>
/// <param name="ThreadsX">Block size, x.</param>
/// <param name="ThreadsY">Block size, y.</param>
/// <param name="ThreadsZ">Block size, z.</param>
/// <param name="SharedBytes">Dynamic shared memory per block (bytes), passed at launch.</param>
internal sealed record CudaKernel(nint Module, nint Function, uint ThreadsX, uint ThreadsY, uint ThreadsZ, int SharedBytes)
{
    /// <summary>The library GEMM the launch runs instead of a kernel, or null for a kernel.</summary>
    public CudaLibraryGemm? Library { get; init; }

    /// <summary>Tensor maps the launch arguments end with, encoded from the pointer arguments (empty for most kernels).</summary>
    public CudaTensorMapSpec[] TensorMaps { get; init; } = [];

    /// <summary>
    /// True when the kernel waits for the previous launch's completion itself (<c>griddepcontrol.wait</c> at its entry, see
    /// <see cref="CudaPtx.DependentHeader"/>), so it may be launched before that launch ends (<see cref="DeviceFeatures.DependentLaunches"/>).
    /// </summary>
    public bool WaitsForPrevious { get; init; }

    /// <summary>Blocks of the kernel the GPU holds at once (occupancy times multiprocessors), 0 when unknown.</summary>
    public int ResidentBlocks { get; init; }

    /// <summary>
    /// Shared memory (static and dynamic) of the blocks of the kernel one multiprocessor holds at once, in bytes: the shared memory
    /// share of the L1 / shared split the multiprocessor needs to run it at full occupancy. A programmatic dependent's blocks are placed
    /// while the previous launch still holds the multiprocessors, only where that launch's split leaves room for them (see
    /// <see cref="CudaSegment"/>'s dependent launches).
    /// </summary>
    public long SharedPerMultiprocessor { get; init; }
}
