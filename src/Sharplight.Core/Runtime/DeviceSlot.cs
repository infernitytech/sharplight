namespace Sharplight.Runtime;

/// <summary>A device participating in a placement, with its share of the model.</summary>
public sealed class DeviceSlot
{
    public required IComputeDevice Device { get; init; }
    /// <summary>Relative share of layers (like llama.cpp's tensor split). Null = proportional to free memory.</summary>
    public double? Weight { get; init; }
    /// <summary>Hard memory cap for this placement (null = device available memory).</summary>
    public long? MemoryLimit { get; init; }
}
