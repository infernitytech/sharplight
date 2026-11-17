using Sharplight.Llm.Serving.Hosting;
using Sharplight.Runtime;

namespace Sharplight.Hosting;

/// <summary>
/// The memory of an opened device as a pool of a <see cref="ModelHost"/>: the id is <c>provider:index</c>, the same as the host's own
/// pools, so every model on the device is charged against one budget.
/// </summary>
/// <param name="device">The opened device.</param>
public sealed class ComputeDeviceMemoryDomain(IComputeDevice device) : IMemoryDomain
{
    /// <summary>The device.</summary>
    public IComputeDevice Device => device;

    /// <inheritdoc/>
    public string Id { get; } = $"{device.Info.Provider}:{device.Info.Index}";

    /// <inheritdoc/>
    public string Name => device.Info.Name;

    /// <inheritdoc/>
    public DeviceMemoryInfo GetMemoryInfo() => device.GetMemoryInfo();
}
