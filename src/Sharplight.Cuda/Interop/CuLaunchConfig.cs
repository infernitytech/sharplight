using System.Runtime.InteropServices;

namespace Sharplight.Cuda.Interop;

/// <summary><c>CUlaunchConfig</c> of <see cref="CudaApi.cuLaunchKernelEx"/>: grid, block, dynamic shared memory, stream and attributes.</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct CuLaunchConfig
{
    public uint GridX;
    public uint GridY;
    public uint GridZ;
    public uint BlockX;
    public uint BlockY;
    public uint BlockZ;
    public uint SharedMemoryBytes;
    public nint Stream;
    public CuLaunchAttribute* Attributes;
    public uint AttributeCount;
}
