using System.Runtime.InteropServices;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

internal static class Rnd
{
    public static float[] F(long n, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (long i = 0; i < n; i++)
        {
            a[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        }

        return a;
    }

    /// <summary>
    /// Values k·s with integer k in [-127, 127] and a power-of-two scale s that differs between 32-value blocks; one value
    /// per block is ±127·s, so 8-bit block quantization (the integer-dot kernels) represents every value exactly.
    /// </summary>
    public static float[] ExactInt8(long n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (long i = 0; i < n; i++)
        {
            float scale = MathF.ScaleB(1f, -4 - (int)(i / 32 % 5));
            int k = i % 32 == (i / 32) % 32 ? (rng.Next(2) == 0 ? 127 : -127) : rng.Next(-127, 128);
            a[i] = k * scale;
        }

        return a;
    }

    public static float[] Pos(long n, int seed, float lo = 0.1f, float hi = 2f)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (long i = 0; i < n; i++)
        {
            a[i] = lo + (float)rng.NextDouble() * (hi - lo);
        }

        return a;
    }

    public static int[] I(long n, int seed, int max)
    {
        var rng = new Random(seed);
        return [.. Enumerable.Range(0, (int)n).Select(_ => rng.Next(max))];
    }

    private static void PutHalf(byte[] b, long off, float v) => BitConverter.TryWriteBytes(b.AsSpan((int)off, 2), (Half)v);

    /// <summary>Random but valid weight bytes (float formats: uniform values; quantized: random quants with sane f16 scales).</summary>
    public static byte[] WeightBytes(DataType t, long elements, int seed, float scale)
    {
        if (t == DataType.F32)
        {
            return MemoryMarshal.AsBytes(F(elements, seed, scale).AsSpan()).ToArray();
        }

        if (t == DataType.F16)
        {
            return MemoryMarshal.AsBytes([.. F(elements, seed, scale).Select(v => (Half)v)]).ToArray();
        }

        if (t == DataType.BF16)
        {
            return MemoryMarshal.AsBytes([.. F(elements, seed, scale).Select(v => (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16))]).ToArray();
        }

        var bytes = new byte[t.ByteSize(elements)];
        new Random(seed).NextBytes(bytes);
        long blocks = elements / t.BlockSize;
        var rng = new Random(seed + 7);
        for (long i = 0; i < blocks; i++)
        {
            long o = i * t.BlockBytes;
            float d() => (0.002f + (float)rng.NextDouble() * 0.01f) * scale;
            if (t == DataType.Q8_0 || t == DataType.Q4_0 || t == DataType.Q5_0)
            {
                PutHalf(bytes, o, d());
            }
            else if (t == DataType.Q4_1 || t == DataType.Q5_1)
            {
                PutHalf(bytes, o, d());
                PutHalf(bytes, o + 2, d());
            }
            else if (t == DataType.Q4_K || t == DataType.Q5_K)
            {
                PutHalf(bytes, o, d());
                PutHalf(bytes, o + 2, d());
            }
            else if (t == DataType.Q6_K)
            {
                PutHalf(bytes, o + 208, d());
            }
            else if (t == DataType.Q1_0 || t == DataType.Q2_0 || t == DataType.Q2_0_G128 || t == DataType.IQ4_NL || t == DataType.IQ4_XS)
            {
                PutHalf(bytes, o, d());
            }
            else if (t == DataType.Tern1_G128)
            {
                PutHalf(bytes, o + 26, d());
            }
            else
            {
                throw new NotSupportedException(t.Name);
            }
        }
        return bytes;
    }
}
