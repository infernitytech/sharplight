using Sharplight.Metal.Interop;

namespace Sharplight.Metal;

/// <summary>
/// An <c>MTLBuffer</c> in shared storage: on unified memory the CPU writes inputs and weights into <see cref="Contents"/> and reads
/// outputs from it, and the kernels address it through <see cref="GpuAddress"/>. Hazard tracking is off (the segment places barriers).
/// </summary>
internal sealed unsafe class MetalBuffer
{
    internal MetalBuffer(nint handle, long bytes, long offset = 0, bool wrapsHostMemory = false)
    {
        Handle = handle;
        Bytes = bytes;
        Contents = (byte*)ObjCRuntime.Send(handle, Selectors.Contents) + offset;
        GpuAddress = ObjCRuntime.SendUInt64(handle, Selectors.GpuAddress) + (ulong)offset;
        WrapsHostMemory = wrapsHostMemory;
    }

    /// <summary>
    /// True when the buffer wraps existing host memory without a copy (a weight in the mapped model file, see
    /// <see cref="MetalDevice.TryWrap"/>): its bytes are already in place and the memory belongs to the weight source.
    /// </summary>
    public bool WrapsHostMemory { get; }

    /// <summary>The retained <c>MTLBuffer</c>.</summary>
    public nint Handle { get; private set; }

    public long Bytes { get; }

    /// <summary>CPU address of the buffer's data (for a wrapped weight: of the weight within the wrapped pages).</summary>
    public byte* Contents { get; }

    /// <summary>GPU virtual address of the buffer's data, passed to kernels as a pointer argument.</summary>
    public ulong GpuAddress { get; }

    /// <summary>Releases the buffer; the caller ensures no command buffer still uses it.</summary>
    public void Release()
    {
        Foundation.Release(Handle);
        Handle = 0;
    }
}
