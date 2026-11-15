using Sharplight.Llm.Serving.Hosting;
using Sharplight.Runtime;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>The memory of an opened device as a pool of the model host (same id as the host's own pools, so models share one budget).</summary>
internal sealed class DeviceMemoryPool(IComputeDevice device) : IMemoryDomain
{
    /// <inheritdoc/>
    public string Id { get; } = $"{device.Info.Provider}:{device.Info.Index}";

    /// <inheritdoc/>
    public string Name => device.Info.Name;

    /// <inheritdoc/>
    public DeviceMemoryInfo GetMemoryInfo() => device.GetMemoryInfo();
}
