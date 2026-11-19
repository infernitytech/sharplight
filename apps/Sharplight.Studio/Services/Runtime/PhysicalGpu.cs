using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>One physical GPU with the providers that can drive it and the one in use.</summary>
/// <param name="Key">The GPU's identity (<see cref="ComputeDeviceOption.GpuKey"/>).</param>
/// <param name="Name">Device name.</param>
/// <param name="TotalMemory">Device memory in bytes.</param>
/// <param name="Entries">The device entries, one per provider (the one in use first).</param>
/// <param name="Choice">The user's choice for this GPU.</param>
/// <param name="Effective">The entry that runs models when a load names no provider.</param>
/// <param name="Note">Why CUDA is not used although it could be (not available here, driver too old, failed), or null.</param>
internal sealed record PhysicalGpu(string Key, string Name, long TotalMemory, IReadOnlyList<ComputeDeviceOption> Entries, ProviderChoice Choice,
    ComputeDeviceOption Effective, string? Note)
{
    /// <summary>The GPU has more than one provider to choose from.</summary>
    public bool HasChoice => Entries.Count > 1;
}
