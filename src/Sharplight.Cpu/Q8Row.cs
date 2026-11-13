namespace Sharplight.Cpu;

/// <summary>
/// One activation row quantized for the integer-dot matmul (<see cref="IntDot"/>): 8-bit values with one scale per 32
/// values, like ggml's q8_0 activations. <see cref="LaneScales"/> repeats each block scale once per four values (the width
/// of one int32 lane of a <c>vpdpbusd</c> result) so the kernels load it as a vector; <see cref="GroupSums"/> holds
/// scale * sum(q) per 16 values for the offset/min terms of the weight formats.
/// </summary>
internal readonly unsafe struct Q8Row(sbyte* q, float* laneScales, float* groupSums)
{
    /// <summary>The quantized values in [-127, 127], K per row.</summary>
    public readonly sbyte* Q = q;

    /// <summary>Block scale per four values (K / 4).</summary>
    public readonly float* LaneScales = laneScales;

    /// <summary>Block scale times the sum of the quantized values, per 16 values (K / 16).</summary>
    public readonly float* GroupSums = groupSums;
}
