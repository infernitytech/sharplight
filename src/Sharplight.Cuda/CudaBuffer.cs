namespace Sharplight.Cuda;

/// <summary>A device memory allocation (or a view into one) of a <see cref="CudaDevice"/>.</summary>
/// <param name="pointer">The device address (<c>CUdeviceptr</c>).</param>
/// <param name="bytes">Size in bytes.</param>
/// <param name="owned">The buffer owns the allocation (frees it); views share another buffer's memory.</param>
internal sealed class CudaBuffer(ulong pointer, long bytes, bool owned = true)
{
    /// <summary>The device address of the first byte.</summary>
    public ulong Pointer { get; } = pointer;

    public long Bytes { get; } = bytes;

    public bool Owned { get; } = owned;
}
