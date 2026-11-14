using Sharplight.Runtime;

namespace Sharplight.Studio.Services.Fit;

/// <summary>The outcome of <see cref="FitPlanner.Plan"/>: the verdict and the figures behind it.</summary>
internal sealed record FitReport
{
    /// <summary>Whether and how the model fits.</summary>
    public required FitVerdict Verdict { get; init; }

    /// <summary>The footprint the verdict is for.</summary>
    public required ModelFootprint Footprint { get; init; }

    /// <summary>Each target device's part (with the memory unloading frees when the verdict counts on it).</summary>
    public required IReadOnlyList<DeviceFit> Devices { get; init; }

    /// <summary>System memory and the offloaded part, when the verdict depends on CPU offload.</summary>
    public DeviceFit? SystemMemory { get; init; }

    /// <summary>Models to unload first (<see cref="FitVerdict.FitsAfterUnloading"/>), or that the runtime unloads before offloading.</summary>
    public IReadOnlyList<string> Unload { get; init; } = [];

    /// <summary>The runtime unloads <see cref="Unload"/> by itself (they are idle and not pinned).</summary>
    public bool UnloadIsAutomatic { get; init; } = true;

    /// <summary>Share of the weights and state that would run on the CPU, 0..1 (with <see cref="FitVerdict.NeedsOffload"/>).</summary>
    public double CpuShare { get; init; }

    /// <summary>CPU offload is turned on in the settings the report is for.</summary>
    public bool CpuOffload { get; init; }

    /// <summary>It fits on the device without offloading, possibly after the runtime unloads idle models by itself.</summary>
    public bool FitsOnDevice => Verdict == FitVerdict.Fits || (Verdict == FitVerdict.FitsAfterUnloading && UnloadIsAutomatic);

    /// <summary>The model loads as configured (it fits, possibly after the runtime unloads idle models or offloads as allowed).</summary>
    public bool LoadsAsConfigured => Verdict switch
    {
        FitVerdict.Fits => true,
        FitVerdict.FitsAfterUnloading => UnloadIsAutomatic,
        FitVerdict.NeedsOffload => CpuOffload,
        _ => false,
    };
}
