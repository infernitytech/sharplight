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
    /// True when the kernel runs work before its <c>griddepcontrol.wait</c> that no earlier launch feeds (<see cref="CudaPtx.PrologueHeader"/>):
    /// it gains from an early launch whatever its blocks' placement, so <see cref="CudaSegment"/> launches it as a dependent regardless
    /// of <see cref="SharedPerBlock"/>.
    /// </summary>
    public bool HasIndependentPrologue { get; init; }

    /// <summary>Blocks of the kernel one multiprocessor holds at once (its occupancy), 0 when unknown.</summary>
    public int BlocksPerMultiprocessor { get; init; }

    /// <summary>
    /// Shared memory one block takes on a multiprocessor (bytes): its static and dynamic shared memory plus the driver's reservation per
    /// block, 0 for a kernel without shared memory. <see cref="CudaSegment"/> launches a kernel as a programmatic dependent only where
    /// its blocks fit the L1 / shared split the previous kernel's blocks set (see <see cref="CudaSegment"/>'s dependent launches).
    /// </summary>
    public long SharedPerBlock { get; init; }
}
