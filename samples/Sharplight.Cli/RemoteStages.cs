using System.Globalization;
using Sharplight.Distributed;
using Sharplight.Runtime;

namespace Sharplight.Cli;

/// <summary>The pipeline stages of a multi-node run (<c>--workers</c>): the local devices first, then one stage per segment worker.</summary>
internal static class RemoteStages
{
    /// <summary>
    /// Connects to the workers and returns the device slots in pipeline order. With one <c>--split</c> share per stage the shares apply;
    /// otherwise every stage takes a share by its free memory.
    /// </summary>
    /// <exception cref="SegmentWorkerException">A worker cannot be reached.</exception>
    public static List<DeviceSlot> Place(IReadOnlyList<IComputeDevice> local, IReadOnlyList<string> workers, IReadOnlyList<double> split, long? localMemoryLimit)
    {
        using var provider = new RemoteExecutionProvider(new RemoteSegmentOptions { Workers = [.. workers] });
        var remote = provider.OpenAll();
        var devices = local.Concat(remote).ToList();
        bool weighted = split.Count == devices.Count;
        var slots = devices.Select((device, i) => new DeviceSlot
        {
            Device = device,
            Weight = weighted ? split[i] : null,
            MemoryLimit = i < local.Count ? localMemoryLimit : null,
        }).ToList();
        foreach (var device in remote)
        {
            Console.Error.WriteLine($"stage {devices.IndexOf(device)}: {device.Info.Name}, {device.GetMemoryInfo().Available >> 20} MiB free, model {device.Worker.ModelName}");
        }

        return slots;
    }

    /// <summary>Prints the per-worker execution figures of a run (round trip, worker time, what the network added).</summary>
    public static void PrintStatistics(IEnumerable<DeviceSlot> slots, TextWriter output)
    {
        foreach (var device in slots.Select(s => s.Device).OfType<RemoteComputeDevice>())
        {
            var s = device.Statistics;
            if (s.Executions == 0)
            {
                continue;
            }

            output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"worker {device.Endpoint}: {s.Executions} executions, round trip {s.RoundTripMs / s.Executions:F3} ms, worker {s.WorkerMs / s.Executions:F3} ms, "
                + $"overhead {s.OverheadMsPerExecution:F3} ms per execution, {s.BytesSent / 1e6:F1} MB sent, {s.BytesReceived / 1e6:F1} MB received"));
        }
    }
}
