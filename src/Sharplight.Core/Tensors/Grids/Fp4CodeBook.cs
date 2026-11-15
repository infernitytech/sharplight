namespace Sharplight.Tensors.Grids;

/// <summary>The 4-bit float code book of <see cref="DataType.FP4_B64"/>: 16 values normalized to [-1, 1], in code order.</summary>
/// <remarks>
/// The f32 values bitsandbytes stores as the <c>quant_map</c> of FP4 weights (bitsandbytes/functional.py <c>get_4bit_type("fp4")</c>,
/// MIT License): the magnitudes 0, 0.0625, 8, 12, 4, 6, 2, 3 divided by 12, then the same with the sign bit set (code 8 is +0 as bitsandbytes builds the table).
/// </remarks>
internal static class Fp4CodeBook
{
    /// <summary>The 16 code values.</summary>
    public static ReadOnlySpan<float> Values =>
    [
        0.0f, 0.0052083334885537624f, 0.6666666865348816f, 1.0f, 0.3333333432674408f, 0.5f, 0.1666666716337204f, 0.25f,
        0.0f, -0.0052083334885537624f, -0.6666666865348816f, -1.0f, -0.3333333432674408f, -0.5f, -0.1666666716337204f, -0.25f,
    ];
}
