using Sharplight.Runtime;

namespace Sharplight.Hosting;

/// <summary>A device a host can place models on.</summary>
/// <param name="Id">Stable id, <c>provider:index</c> (the memory pool id the model host charges).</param>
/// <param name="Name">Device name.</param>
/// <param name="Provider">The execution provider.</param>
/// <param name="Info">The provider's description of the device.</param>
public sealed record HostDevice(string Id, string Name, IExecutionProvider Provider, DeviceInfo Info)
{
    /// <summary>A GPU (else the CPU, whose memory is the system memory).</summary>
    public bool IsGpu => Info.Kind is not DeviceKind.Cpu;

    /// <summary>Memory the device has, in bytes (0 = unknown).</summary>
    public long TotalBytes => Info.TotalMemory;
}
