using Sharplight.Runtime;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary><see cref="IMemoryDomain"/> over an opened compute device (live budget from the driver when it has one).</summary>
public sealed class DeviceMemoryDomain(IComputeDevice device) : IMemoryDomain
{
    public IComputeDevice Device { get; } = device;
    public string Id { get; } = $"{device.Info.Provider}:{device.Info.Index}";
    public string Name => Device.Info.Name;
    public DeviceMemoryInfo GetMemoryInfo() => Device.GetMemoryInfo();
}
