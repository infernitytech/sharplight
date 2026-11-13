namespace Sharplight.Runtime;

/// <summary>Result of placement: which device each group/node goes to and the resulting segments.</summary>
public sealed class PlacementPlan
{
    /// <summary>Devices in use: the pipeline devices in order, followed by the fallback device when one was given.</summary>
    public required IReadOnlyList<DeviceSlot> Slots { get; init; }
    /// <summary>The graph cut into consecutive device-local segments, in execution order.</summary>
    public required IReadOnlyList<GraphSegment> Segments { get; init; }
    /// <summary>Index into <see cref="Slots"/> per placement group (<see cref="Graph.Node.Group"/>).</summary>
    public required IReadOnlyDictionary<string, int> GroupDevice { get; init; }
    /// <summary>Bytes of weights and state assigned to each slot.</summary>
    public required IReadOnlyDictionary<int, long> BytesPerSlot { get; init; }

    /// <summary>Multi-line, human-readable summary of the placement.</summary>
    public string Describe()
    {
        var lines = new List<string>();
        for (int s = 0; s < Slots.Count; s++)
        {
            var groups = GroupDevice.Where(kv => kv.Value == s).Select(kv => kv.Key).ToList();
            lines.Add($"slot {s} {Slots[s].Device.Info.Name}: {groups.Count} groups, {BytesPerSlot.GetValueOrDefault(s) / (1024.0 * 1024 * 1024):F2} GiB weights+state");
        }
        lines.Add($"{Segments.Count} segment(s): " + string.Join(" → ", Segments.Select(sg => $"[{sg.Index}@{sg.DeviceSlot}: {sg.Nodes.Count} nodes]")));
        return string.Join(Environment.NewLine, lines);
    }
}
