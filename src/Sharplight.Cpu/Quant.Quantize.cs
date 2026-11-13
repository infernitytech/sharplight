namespace Sharplight.Cpu;

// Quantization of f32 rows into the GGUF block formats used for quantized KV caches (llama.cpp's reference quantizers).
internal static unsafe partial class Quant
{
    /// <summary>Whether <see cref="Quantize"/> supports the format.</summary>
    public static bool CanQuantize(WeightFormat f) => f is WeightFormat.Q8_0 or WeightFormat.Q4_0;

    /// <summary>
    /// Quantizes <paramref name="count"/> values (a multiple of 32) into blocks at <paramref name="dst"/>: q8_0 scales by
    /// amax / 127 and rounds to nearest; q4_0 scales by the value of largest magnitude / -8 and stores q + 8 as nibbles
    /// (value j of a block in the low nibble of byte j, value j + 16 in its high nibble).
    /// </summary>
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
            else
            {
                throw new NotSupportedException($"Quantizing to {f}");
            }
        }
    }

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
}
