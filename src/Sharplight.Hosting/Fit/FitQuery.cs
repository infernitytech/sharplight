using Sharplight.Runtime;

namespace Sharplight.Hosting.Fit;

/// <summary>What <see cref="FitPlanner"/> decides on: a footprint, the devices it would load on and what else holds their memory.</summary>
public sealed record FitQuery
{
    /// <summary>The estimated footprint of the model with its settings.</summary>
    public required ModelFootprint Footprint { get; init; }

    /// <summary>The devices the model loads on, in pipeline order (the first takes the offloaded part's place); one unless split.</summary>
    public required IReadOnlyList<DeviceMemory> Targets { get; init; }

    /// <summary>The CPU's system memory: where offloaded layers go. Null when there is no CPU device.</summary>
    public DeviceMemory? SystemMemory { get; init; }

    /// <summary>Memory kept free on every device for other applications.</summary>
    public long ReserveBytes { get; init; }

    /// <summary>The resident models.</summary>
    public IReadOnlyList<ResidentUsage> Residents { get; init; } = [];

    /// <summary>CPU offload is turned on for this load.</summary>
    public bool CpuOffload { get; init; }

    /// <summary>
    /// Routed experts may stay on disk (<c>TaskLoadOptions.ExpertsOnDisk</c>): an offloaded part larger than the system memory still loads,
    /// its experts read from disk on demand (assumed to be most of a mixture-of-experts model's weights; the engine decides at load).
    /// </summary>
    public bool ExpertsOnDisk { get; init; }
}
