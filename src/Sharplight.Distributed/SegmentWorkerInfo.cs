using Sharplight.Distributed.Protocol;
using Sharplight.Distributed.Transport;
using Sharplight.Runtime;

namespace Sharplight.Distributed;

/// <summary>What a segment worker reported when the coordinator connected: its devices, memory and model.</summary>
public sealed record SegmentWorkerInfo
{
    /// <summary>The worker's name (host name by default).</summary>
    public required string Name { get; init; }

    /// <summary>Random id of the worker process: two connections with the same id reach the same process.</summary>
    public required string InstanceId { get; init; }

    /// <summary>Where neighbouring workers reach this worker, or null for the address the coordinator used.</summary>
    public SegmentEndpoint? AdvertisedEndpoint { get; init; }

    /// <summary>The model the worker loaded (its definition name).</summary>
    public required string ModelName { get; init; }

    /// <summary>The model's architecture.</summary>
    public required string Architecture { get; init; }

    /// <summary>The worker's devices: its pipeline devices in order, then its fallback device when it offloads to the CPU.</summary>
    public required IReadOnlyList<SegmentWorkerDevice> Devices { get; init; }

    /// <summary>Device features enabled on every pipeline device of the worker (<see cref="DeviceFeatures"/> names).</summary>
    public required IReadOnlySet<string> Features { get; init; }

    /// <summary>The worker's Sharplight version.</summary>
    public required string Version { get; init; }

    internal void Write(PayloadWriter w)
    {
        w.String(Name).String(InstanceId).String(AdvertisedEndpoint?.ToString() ?? "").String(ModelName).String(Architecture).String(Version);
        w.Int32(Devices.Count);
        foreach (var d in Devices)
        {
            w.String(d.Provider).String(d.Name).Int32((int)d.Kind).Int64(d.TotalMemory).Int64(d.AvailableMemory).Bool(d.IsFallback);
        }

        w.Strings([.. Features.Order(StringComparer.Ordinal)]);
    }

    internal static SegmentWorkerInfo Read(PayloadReader r)
    {
        string name = r.String();
        string instance = r.String();
        string advertised = r.String();
        string model = r.String();
        string architecture = r.String();
        string version = r.String();
        int count = r.Count();
        var devices = new List<SegmentWorkerDevice>(count);
        for (int i = 0; i < count; i++)
        {
            devices.Add(new SegmentWorkerDevice(r.String(), r.String(), (DeviceKind)r.Int32(), r.Int64(), r.Int64(), r.Bool()));
        }

        return new SegmentWorkerInfo
        {
            Name = name,
            InstanceId = instance,
            AdvertisedEndpoint = advertised.Length == 0 ? null : SegmentEndpoint.Parse(advertised),
            ModelName = model,
            Architecture = architecture,
            Version = version,
            Devices = devices,
            Features = r.Strings().ToHashSet(StringComparer.Ordinal),
        };
    }
}
