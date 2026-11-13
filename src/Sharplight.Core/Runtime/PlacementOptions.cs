namespace Sharplight.Runtime;

public sealed class PlacementOptions
{
    /// <summary>Devices in pipeline order (first device runs the first layers).</summary>
    public required IReadOnlyList<DeviceSlot> Devices { get; init; }
    /// <summary>Fallback device for unsupported ops and layers that do not fit (CPU offload). Optional.</summary>
    public DeviceSlot? Fallback { get; init; }
    /// <summary>Bytes reserved on every device for activations, scratch and driver overhead.</summary>
    public long ReservePerDevice { get; init; } = 1L << 30;
    public required CompileOptions Compile { get; init; }
}
