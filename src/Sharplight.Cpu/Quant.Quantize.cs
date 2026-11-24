using System.Runtime.CompilerServices;
namespace Sharplight.Cpu;

// Quantization of f32 rows into the block formats used for quantized KV caches (the GGUF formats with llama.cpp's reference
// quantizers, and block-scaled FP8).
internal static unsafe partial class Quant
{
    /// <summary>Whether <see cref="Quantize"/> supports the format.</summary>
    public static bool CanQuantize(WeightFormat f) => f is WeightFormat.Q8_0 or WeightFormat.Q4_0 or WeightFormat.F8E4M3_B32;

    /// <summary>
    /// Quantizes <paramref name="count"/> values (a multiple of 32) into blocks at <paramref name="dst"/>: q8_0 scales by
    /// amax / 127 and rounds to nearest; q4_0 scales by the value of largest magnitude / -8 and stores q + 8 as nibbles
    /// (value j of a block in the low nibble of byte j, value j + 16 in its high nibble); f8e4m3_b32 scales by amax / 448 (an f32
    /// scale) and stores E4M3 codes rounded to nearest even.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Quantize(WeightFormat f, float* src, long count, byte* dst)
    {
        long blocks = count / 32;
        for (long b = 0; b < blocks; b++, src += 32)
        {
            if (f == WeightFormat.Q8_0)
            {
                QuantizeQ8_0(src, dst + b * 34);
            }
            else if (f == WeightFormat.Q4_0)
            {
                QuantizeQ4_0(src, dst + b * 18);
            }
            else if (f == WeightFormat.F8E4M3_B32)
            {
                QuantizeF8E4M3_B32(src, dst + b * 36);
            }
            else
            {
                throw new NotSupportedException($"Quantizing to {f}");
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void QuantizeQ8_0(float* x, byte* dst)
    {
        float amax = 0f;
        for (int i = 0; i < 32; i++)
        {
            amax = MathF.Max(amax, MathF.Abs(x[i]));
        }

        float d = amax / 127f;
        float inv = d != 0f ? 1f / d : 0f;
        *(ushort*)dst = Simd.FloatToHalf(d);
        sbyte* q = (sbyte*)(dst + 2);
        for (int i = 0; i < 32; i++)
        {
            q[i] = (sbyte)MathF.Round(x[i] * inv, MidpointRounding.AwayFromZero);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void QuantizeQ4_0(float* x, byte* dst)
    {
        float amax = 0f;
        float max = 0f;
        for (int i = 0; i < 32; i++)
        {
            if (MathF.Abs(x[i]) > amax)
            {
                amax = MathF.Abs(x[i]);
                max = x[i];
            }
        }

        float d = max / -8f;
        float inv = d != 0f ? 1f / d : 0f;
        *(ushort*)dst = Simd.FloatToHalf(d);
        byte* q = dst + 2;
        for (int i = 0; i < 16; i++)
        {
            int lo = Math.Min(15, (int)(x[i] * inv + 8.5f));
            int hi = Math.Min(15, (int)(x[i + 16] * inv + 8.5f));
            q[i] = (byte)(lo | (hi << 4));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void QuantizeF8E4M3_B32(float* x, byte* dst)
    {
        float amax = 0f;
        for (int i = 0; i < 32; i++)
        {
            amax = MathF.Max(amax, MathF.Abs(x[i]));
        }

        float d = amax / 448f;
        float inv = d != 0f ? 1f / d : 0f;
        *(float*)dst = d;
        for (int i = 0; i < 32; i++)
        {
            dst[4 + i] = FloatToE4m3(x[i] * inv);
        }
    }

    /// <summary>
    /// The E4M3FN code of <paramref name="v"/>, rounded to nearest even and saturated at ±448 (subnormals below 2^-6 as multiples of
    /// 2^-9); the encoder of the KV-cache write kernels (e4m3.slang).
    /// </summary>
    internal static byte FloatToE4m3(float v)
    {
        uint sign = v < 0f ? 0x80u : 0u;
        float a = MathF.Abs(v);
        uint code;
        if (a < 0.015625f)
        {
            code = (uint)MathF.Round(a * 512f, MidpointRounding.AwayFromZero);
        }
        else
        {
            uint bits = BitConverter.SingleToUInt32Bits(a);
            bits += 0x7FFFFu + ((bits >> 20) & 1u);
            code = Math.Min((bits >> 20) - (120u << 3), 0x7Eu);
        }

        return (byte)(sign | code);
    }
}
