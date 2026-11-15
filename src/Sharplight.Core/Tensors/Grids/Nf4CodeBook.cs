namespace Sharplight.Tensors.Grids;

/// <summary>The NF4 code book of <see cref="DataType.NF4_B64"/>: 16 normal-distribution quantiles normalized to [-1, 1], in code order.</summary>
/// <remarks>The f32 values bitsandbytes stores as the <c>quant_map</c> of NF4 weights (bitsandbytes/functional.py, MIT License).</remarks>
internal static class Nf4CodeBook
{
    /// <summary>The 16 code values.</summary>
    public static ReadOnlySpan<float> Values =>
    [
        -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f, -0.28444138169288635f, -0.18477343022823334f,
        -0.09105003625154495f, 0.0f, 0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
        0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f,
    ];
}
