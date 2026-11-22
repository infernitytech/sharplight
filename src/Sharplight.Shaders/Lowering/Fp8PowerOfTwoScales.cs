using System.Runtime.InteropServices;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Requantization of block-scaled E4M3 values to power-of-two (UE8M0) scales, the operands of the block-scaled MMA
/// (<see cref="WeightRepack.Fp8B128GroupedUe8m0"/>): each block of 128 values with f32 scale d gets the scale
/// d' = 2^ceil(log2(amax / 448)) of its largest value, and each code c becomes the E4M3 code of e4m3(c) * d / d', rounded to nearest
/// even as the kernels' <c>e4m3_code</c> rounds (e4m3.slang). vLLM's DeepGEMM path does the same per 128 x 128 block; here the blocks
/// are 128 values of one row, so rows of a block that do not reach its maximum keep more of their precision.
/// </summary>
internal static class Fp8PowerOfTwoScales
{
    private static readonly float[] s_values = BuildValues();

    /// <summary>The value of every E4M3 code (NaN codes as 0: no checkpoint stores them).</summary>
    private static float[] BuildValues()
    {
        var v = new float[256];
        for (int c = 0; c < 256; c++)
        {
            int e = (c >> 3) & 15, m = c & 7;
            float a = (c & 0x7F) == 0x7F ? 0f : e == 0 ? m / 512f : (1f + m / 8f) * MathF.Pow(2f, e - 7);
            v[c] = (c & 0x80) != 0 ? -a : a;
        }

        return v;
    }

    /// <summary>The value of E4M3 code <paramref name="code"/>.</summary>
    internal static float Value(byte code) => s_values[code];

    /// <summary>The E4M3 code of <paramref name="v"/> (|v| &lt;= 448), rounded to nearest even like <c>e4m3_code</c> in e4m3.slang.</summary>
    internal static byte Code(float v)
    {
        uint sign = v < 0 ? 0x80u : 0u;
        float a = MathF.Abs(v);
        uint code;
        if (a < 0.015625f)
        {
            code = (uint)MathF.Round(a * 512f, MidpointRounding.ToEven);
        }
        else
        {
            uint bits = BitConverter.SingleToUInt32Bits(a);
            bits += 0x7FFFFu + ((bits >> 20) & 1u);
            code = Math.Min((bits >> 20) - (120u << 3), 0x7Eu);
        }

        return (byte)(sign | code);
    }

    /// <summary>The power-of-two scale for a block whose largest magnitude is <paramref name="amax"/> (2^ceil(log2(amax / 448))).</summary>
    internal static float Scale(float amax)
    {
        if (!(amax > 0) || !float.IsFinite(amax))
        {
            return 1f;
        }

        uint bits = BitConverter.SingleToUInt32Bits(amax / 448f);
        uint exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0 ? 1u : 0u);
        return BitConverter.UInt32BitsToSingle(Math.Clamp(exponent, 1u, 254u) << 23);
    }

    /// <summary>
    /// Requantizes one unit of the grouped layout in place: [<paramref name="rows"/> rows of <paramref name="blocks"/> x 128 codes]
    /// followed by [<paramref name="blocks"/>][<paramref name="rows"/>] f32 scales.
    /// </summary>
    internal static void Requantize(Span<byte> unit, int blocks, int rows)
    {
        var codes = unit[..(rows * blocks * 128)];
        var scales = MemoryMarshal.Cast<byte, float>(unit[(rows * blocks * 128)..]);
        for (int row = 0; row < rows; row++)
        {
            for (int b = 0; b < blocks; b++)
            {
                RequantizeBlock(codes.Slice((row * blocks + b) * 128, 128), ref scales[b * rows + row]);
            }
        }
    }

    /// <summary>Requantizes the 128 codes of one block with scale <paramref name="scale"/> to its power-of-two scale (updated).</summary>
    internal static void RequantizeBlock(Span<byte> block, ref float scale)
    {
        float m = 0;
        foreach (byte c in block)
        {
            m = MathF.Max(m, MathF.Abs(s_values[c]));
        }

        float d = Scale(m * scale);
        float factor = scale / d;
        if (factor != 1f)
        {
            for (int i = 0; i < block.Length; i++)
            {
                block[i] = Code(s_values[block[i]] * factor);
            }
        }

        scale = d;
    }
}
