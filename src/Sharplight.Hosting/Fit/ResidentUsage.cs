namespace Sharplight.Hosting.Fit;

/// <summary>Memory a resident model holds, and whether the runtime may unload it by itself to make room.</summary>
/// <param name="Id">The model's id.</param>
/// <param name="DeviceIds">The physical devices it is spread over (its memory counts evenly on each).</param>
/// <param name="Bytes">Device memory it holds.</param>
/// <param name="CanEvict">Idle and not pinned: the runtime unloads it when another model needs the room.</param>
/// <param name="LastUsed">When it last served a request (least recently used ones go first).</param>
public sealed record ResidentUsage(string Id, IReadOnlyList<string> DeviceIds, long Bytes, bool CanEvict, DateTimeOffset? LastUsed);
