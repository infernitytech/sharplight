using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>No per-device data yet: Automatic means Vulkan (docs/cuda.md, phase 1 changes no default).</summary>
internal sealed class NoProviderHints : IProviderPerformanceHints
{
    /// <inheritdoc/>
    public string? FasterProvider(IReadOnlyList<ComputeDeviceOption> gpu) => null;
}
