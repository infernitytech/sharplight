namespace Sharplight.Runtime;

/// <summary>Result of placement: which device each group/node goes to and the resulting segments.</summary>
public sealed class PlacementPlan
{
    public required IReadOnlyList<DeviceSlot> Slots { get; init; }
    public required IReadOnlyList<GraphSegment> Segments { get; init; }
    public required IReadOnlyDictionary<string, int> GroupDevice { get; init; }
    public required IReadOnlyDictionary<int, long> BytesPerSlot { get; init; }

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
