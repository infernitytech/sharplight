using System.Runtime.InteropServices;

namespace Sharplight.Cuda.Interop;

/// <summary>
/// <c>CUlaunchAttribute</c>: an attribute id (padded to 8 bytes) and its 64-byte value union, of which the provider sets the first 32-bit
/// field (<c>programmaticStreamSerializationAllowed</c>).
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = 72)]
internal struct CuLaunchAttribute
{
    [FieldOffset(0)]
    public int Id;

    [FieldOffset(8)]
    public int Value;
}
