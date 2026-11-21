using System.Runtime.InteropServices;

namespace Sharplight.Cuda.Interop;

/// <summary><c>CUmemcpyAttributes</c>: the source access order, two location hints (type, id; ignored for pinned host and device memory) and flags.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct MemcpyAttributes
{
    public int SourceAccessOrder;
    public int SourceLocationType;
    public int SourceLocationId;
    public int DestinationLocationType;
    public int DestinationLocationId;
    public uint Flags;
}
