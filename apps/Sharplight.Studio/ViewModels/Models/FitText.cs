using Sharplight.Hosting.Fit;
using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Fit;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>Wording of fit estimates, shared by the load panel and the download badges.</summary>
internal static class FitText
{
    /// <summary>
    /// The verdict as a heading: <c>Fits on the GPU</c> (or <c>in system memory</c>, <c>on 2 GPUs</c>), <c>Fits after unloading other models</c>,
    /// <c>Fits with CPU offload</c> (<c>CPU expert offload</c> for a mixture-of-experts model whose offload moves experts only; <c>Needs</c>
    /// while offload is off) or <c>Does not fit</c>.
    /// </summary>
    /// <param name="report">The plan.</param>
    /// <param name="expertOffload">The offload moves routed experts only (<see cref="FitAdvisor.IsExpertOffload"/>).</param>
    public static string Title(FitReport report, bool expertOffload = false) => report.Verdict switch
    {
        FitVerdict.Fits => report.Devices.Count switch
        {
            1 when !report.Devices[0].Device.IsGpu => "Fits in system memory",
            var count and > 1 => $"Fits on {count} GPUs",
            _ => "Fits on the GPU",
        },
        FitVerdict.FitsAfterUnloading => "Fits after unloading other models",
        FitVerdict.NeedsOffload => $"{(report.CpuOffload ? "Fits with" : "Needs")} {(expertOffload ? "CPU expert offload" : "CPU offload")}",
        _ => "Does not fit",
    };

    /// <summary>Severity: it loads as configured (counting automatic unloads).</summary>
    /// <param name="report">The plan.</param>
    public static bool IsGood(FitReport report) => report.FitsOnDevice;

    /// <summary>Severity: it does not fit at all.</summary>
    /// <param name="report">The plan.</param>
    public static bool IsBad(FitReport report) => report.Verdict == FitVerdict.DoesNotFit;

    /// <summary><c>Weights 7.6 GB · KV cache 1.1 GB · activations 0.4 GB</c>.</summary>
    /// <param name="footprint">The footprint.</param>
    public static string Breakdown(ModelFootprint footprint)
        => $"Weights {ByteFormat.Format(footprint.WeightBytes)} · KV cache {ByteFormat.Format(footprint.StateBytes)} · activations {ByteFormat.Format(footprint.ActivationBytes)}";

    /// <summary>What the verdict means, in one or two sentences.</summary>
    /// <param name="report">The plan.</param>
    /// <param name="target">Where it loads, e.g. a device name or <c>2 GPUs</c>.</param>
    /// <param name="expertOffload">The offload moves routed experts only (<see cref="FitAdvisor.IsExpertOffload"/>).</param>
    public static string Message(FitReport report, string target, bool expertOffload = false)
    {
        string total = ByteFormat.Format(report.Footprint.Total);
        long available = report.Devices.Sum(device => device.AvailableBytes);
        string names = Names(report.Unload);
        switch (report.Verdict)
        {
            case FitVerdict.Fits:
                return $"Needs about {total} of the {ByteFormat.Format(available)} available on {target}.";
            case FitVerdict.FitsAfterUnloading:
                return report.UnloadIsAutomatic
                    ? $"Needs about {total}. Fits once {names} {(report.Unload.Count == 1 ? "is" : "are")} unloaded; idle models unload automatically when it loads."
                    : $"Needs about {total}. Unload {names} first: {(report.Unload.Count == 1 ? "it is" : "they are")} pinned or in use.";
            case FitVerdict.NeedsOffload:
                var offload = expertOffload
                    ? $"Needs about {total}; {target} has {ByteFormat.Format(available)} available. The experts of some layers (about {FitAdvisor.Percent(report.CpuShare)} of the model) "
                      + "would run on the CPU, attention and the shared layers stay on the GPU; slower than all on the GPU."
                    : $"Needs about {total}; {target} has {ByteFormat.Format(available)} available. About {FitAdvisor.Percent(report.CpuShare)} of the model would run on the CPU, which is slower.";
                if (report.Unload.Count > 0)
                {
                    offload += $" Idle models unload first: {names}.";
                }

                if (report.ExpertsFromDisk)
                {
                    offload += $" The part left over is more than the system memory: its routed experts are read from disk on demand, so expect about 1 token/s or less.";
                }

                return report.CpuOffload ? offload : offload + " Turn on CPU offload, or the load fails.";
            default:
                if (report.SystemMemory is { } system)
                {
                    return $"Needs about {total}; {target} has {ByteFormat.Format(available)} available, and the {ByteFormat.Format(system.NeededBytes)} left over "
                        + $"is more than the {ByteFormat.Format(system.AvailableBytes)} of system memory available.";
                }

                return report.Devices.Any(device => device.Device.IsGpu)
                    ? $"Needs about {total}; {target} has {ByteFormat.Format(available)} available, even with every other model unloaded."
                    : $"Needs about {total}; system memory has {ByteFormat.Format(available)} available, even with every other model unloaded.";
        }
    }

    /// <summary>Where a plan loads the model: the device's name, or <c>2 GPUs</c>.</summary>
    /// <param name="report">The plan.</param>
    public static string Target(FitReport report) => report.Devices.Count switch
    {
        0 => "the device",
        1 => report.Devices[0].Device.IsGpu ? report.Devices[0].Device.Name : "system memory",
        var count => $"{count} GPUs",
    };

    /// <summary><c>a</c>, <c>a and b</c>, <c>a, b and c</c>.</summary>
    private static string Names(IReadOnlyList<string> names) => names.Count switch
    {
        0 => "",
        1 => names[0],
        _ => string.Join(", ", names.Take(names.Count - 1)) + " and " + names[^1],
    };
}
