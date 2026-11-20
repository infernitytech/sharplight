namespace Sharplight.Runtime;

/// <summary>How a graph is distributed over devices and compiled.</summary>
public sealed class PlacementOptions
{
    /// <summary>Devices in pipeline order (first device runs the first layers).</summary>
    public required IReadOnlyList<DeviceSlot> Devices { get; init; }
    /// <summary>Fallback device for unsupported ops and layers that do not fit (CPU offload). Optional.</summary>
    public DeviceSlot? Fallback { get; init; }
    /// <summary>Bytes reserved on every device for activations, scratch and driver overhead.</summary>
    public long ReservePerDevice { get; init; } = 1L << 30;
    /// <summary>
    /// Bytes reserved on a device for every extra segment that moving routed experts to the <see cref="Fallback"/> creates
    /// (<see cref="SplitExperts"/>, <see cref="StreamExpertsMinRows"/>): a compiled segment's stream, launch recordings and library workspace.
    /// </summary>
    public long ReservePerSegment { get; init; } = 32L << 20;
    /// <summary>Bounds and feature switches used to compile every segment.</summary>
    public required CompileOptions Compile { get; init; }
    /// <summary>
    /// Whether nodes are ordered by data-dependent count level, so that the independent counts a device computes (e.g. the
    /// per-class NonZero and NonMaxSuppression of a detection head) share one segment and one round trip to the host instead of
    /// ending a segment each. Default true; <c>SHARPLIGHT_COUNT_LEVELS=0</c> also turns it off.
    /// </summary>
    public bool GroupDataDependentCounts { get; init; } = true;
    /// <summary>
    /// With a <see cref="Fallback"/> device, whether the routed experts of mixture-of-experts layers are placed apart from the rest of their
    /// layer: when the model does not fit, the routed experts of the first layers go to the fallback (as llama.cpp's <c>--n-cpu-moe</c>), and
    /// attention, shared experts, dense layers and the output head stay on the devices. Default true; <c>SHARPLIGHT_SPLIT_EXPERTS=0</c> also
    /// turns it off (whole layers go to the fallback).
    /// </summary>
    public bool SplitExperts { get; init; } = true;
    /// <summary>
    /// With <see cref="SplitExperts"/> and a CPU fallback: from this many rows per forward pass (prefill) the routed experts that the
    /// placement put on the CPU run on the GPU before them instead, their weights copied in for each pass
    /// (<see cref="CompileOptions.StreamWeights"/>; the GPU keeps room for one layer's experts). 0 turns streaming off. Default 32;
    /// <c>SHARPLIGHT_STREAM_EXPERTS_MIN_ROWS</c> overrides it.
    /// </summary>
    public int StreamExpertsMinRows { get; init; } = int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_STREAM_EXPERTS_MIN_ROWS"), out int rows) && rows >= 0 ? rows : 32;

    /// <summary>A copy with the compile options <paramref name="compile"/> and every other setting kept.</summary>
    internal PlacementOptions WithCompile(CompileOptions compile) => new()
    {
        Devices = Devices,
        Fallback = Fallback,
        ReservePerDevice = ReservePerDevice,
        ReservePerSegment = ReservePerSegment,
        Compile = compile,
        GroupDataDependentCounts = GroupDataDependentCounts,
        SplitExperts = SplitExperts,
        StreamExpertsMinRows = StreamExpertsMinRows,
    };
}
