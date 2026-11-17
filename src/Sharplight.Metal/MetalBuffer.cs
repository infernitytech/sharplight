using Sharplight.Metal.Interop;

namespace Sharplight.Metal;

/// <summary>
/// An <c>MTLBuffer</c> in shared storage: on unified memory the CPU writes inputs and weights into <see cref="Contents"/> and reads
/// outputs from it, and the kernels address it through <see cref="GpuAddress"/>. Hazard tracking is off (the segment places barriers).
/// </summary>
internal sealed unsafe class MetalBuffer
{
    internal MetalBuffer(nint handle, long bytes)
    {
        Handle = handle;
        Bytes = bytes;
        Contents = (byte*)ObjCRuntime.Send(handle, Selectors.Contents);
        GpuAddress = ObjCRuntime.SendUInt64(handle, Selectors.GpuAddress);
    }

    /// <summary>The retained <c>MTLBuffer</c>.</summary>
    public nint Handle { get; private set; }

    public long Bytes { get; }

    /// <summary>CPU address of the buffer's memory.</summary>
    public byte* Contents { get; }

    /// <summary>GPU virtual address of the buffer (Metal 3), passed to kernels as a pointer argument.</summary>
    public ulong GpuAddress { get; }

    /// <summary>Releases the buffer; the caller ensures no command buffer still uses it.</summary>
    public void Release()
    {
        Foundation.Release(Handle);
        Handle = 0;
    }
}
