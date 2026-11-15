using System.Buffers.Binary;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// The group-scaled weight formats of quantized Hugging Face checkpoints (q4z_g32 / q4z_g128, q8z_g32 / q8z_g128, nf4_b64,
/// nvfp4_gs) against a scalar decoding of their documented layouts, on random blocks. The importer-level checks against the
/// libraries' own dequantization are in Sharplight.Import.HuggingFace.Tests.QuantizedImportTests.
/// </summary>
public unsafe class GroupedDequantTests
{
    private static readonly float[] Fp4 = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];

    [Theory]
    [InlineData("q4z_g32", 32)]
    [InlineData("q4z_g128", 128)]
    public void Q4z(string name, int group)
    {
        var rng = new Random(group);
        int blocks = 5;
        var bytes = new byte[blocks * (4 + group / 2)];
        rng.NextBytes(bytes);
        var expected = new float[blocks * group];
        for (int b = 0; b < blocks; b++)
        {
            var block = bytes.AsSpan(b * (4 + group / 2));
            float d = WriteHalf(block, (float)(rng.NextDouble() * 0.02 + 1e-4));
            float z = WriteHalf(block[2..], rng.Next(16));
            for (int i = 0; i < group; i++)
            {
                int run = i / 32;
                int j = i % 32;
                int code = (block[4 + 16 * run + (j & 15)] >> (j < 16 ? 0 : 4)) & 0xF;
                expected[b * group + i] = (code - z) * d;
            }
        }

        Assert.Equal(expected, Decode(DataType.Get(name), bytes));
    }

    [Theory]
    [InlineData("q8z_g32", 32)]
    [InlineData("q8z_g128", 128)]
    public void Q8z(string name, int group)
    {
        var rng = new Random(group + 1);
        int blocks = 5;
        var bytes = new byte[blocks * (4 + group)];
        rng.NextBytes(bytes);
        var expected = new float[blocks * group];
        for (int b = 0; b < blocks; b++)
        {
            var block = bytes.AsSpan(b * (4 + group));
            float d = WriteHalf(block, (float)(rng.NextDouble() * 0.002 + 1e-5));
            float z = WriteHalf(block[2..], rng.Next(256));
            for (int i = 0; i < group; i++)
            {
                expected[b * group + i] = (block[4 + i] - z) * d;
            }
        }

        Assert.Equal(expected, Decode(DataType.Get(name), bytes));
    }

    [Fact]
    public void Nf4()
    {
        var rng = new Random(7);
        int blocks = 4;
        var bytes = new byte[blocks * 36];
        rng.NextBytes(bytes);
        var expected = new float[blocks * 64];
        for (int b = 0; b < blocks; b++)
        {
            var block = bytes.AsSpan(b * 36);
            float d = (float)(rng.NextDouble() * 0.1);
            BinaryPrimitives.WriteSingleLittleEndian(block, d);
            for (int i = 0; i < 64; i++)
            {
                int j = i % 32;
                int code = (block[4 + 16 * (i / 32) + (j & 15)] >> (j < 16 ? 0 : 4)) & 0xF;
                expected[b * 64 + i] = Quant.Nf4Values[code] * d;
            }
        }

        Assert.Equal(expected, Decode(DataType.NF4_B64, bytes));
    }

    [Fact]
    public void Fp4Lookup()
    {
        var rng = new Random(11);
        var bytes = new byte[4 * 36];
        rng.NextBytes(bytes);
        var expected = new float[4 * 64];
        for (int b = 0; b < 4; b++)
        {
            var block = bytes.AsSpan(b * 36);
            float d = (float)(rng.NextDouble() * 0.1);
            BinaryPrimitives.WriteSingleLittleEndian(block, d);
            for (int i = 0; i < 64; i++)
            {
                int j = i % 32;
                int code = (block[4 + 16 * (i / 32) + (j & 15)] >> (j < 16 ? 0 : 4)) & 0xF;
                expected[b * 64 + i] = Quant.Fp4Values[code] * d;
            }
        }

        Assert.Equal(expected, Decode(DataType.FP4_B64, bytes));
    }

    [Fact]
    public void Int8WithF32Absmax()
    {
        var rng = new Random(12);
        var bytes = new byte[4 * 36];
        rng.NextBytes(bytes);
        var expected = new float[4 * 32];
        for (int b = 0; b < 4; b++)
        {
            float d = (float)(rng.NextDouble() * 3);
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(b * 36), d);
            for (int i = 0; i < 32; i++)
            {
                expected[b * 32 + i] = (sbyte)bytes[b * 36 + 4 + i] * d * 7.874015718698502e-3f;
            }
        }

        Assert.Equal(expected, Decode(DataType.Q8A_G32, bytes));
    }

    [Fact]
    public void TernaryWithF32Scale()
    {
        var rng = new Random(13);
        var bytes = new byte[3 * 36];
        rng.NextBytes(bytes);
        var expected = new float[3 * 128];
        for (int b = 0; b < 3; b++)
        {
            float d = (float)(rng.NextDouble() * 3);
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(b * 36), d);
            for (int i = 0; i < 128; i++)
            {
                int code = (bytes[b * 36 + 4 + i % 32] >> (6 - 2 * (i / 32))) & 3;
                expected[b * 128 + i] = (code - 1) * d;
            }
        }

        Assert.Equal(expected, Decode(DataType.Q2S_G128, bytes));
    }

    [Fact]
    public void Nvfp4WithGlobalScale()
    {
        var rng = new Random(9);
        int blocks = 4;
        var bytes = new byte[blocks * 40];
        rng.NextBytes(bytes);
        var expected = new float[blocks * 64];
        for (int b = 0; b < blocks; b++)
        {
            var block = bytes.AsSpan(b * 40);
            float g = (float)(rng.NextDouble() * 0.01 + 1e-4);
            BinaryPrimitives.WriteSingleLittleEndian(block, g);
            for (int s = 0; s < 4; s++)
            {
                block[4 + s] = (byte)rng.Next(0x7F);   // UE4M3 without NaN
                float scale = Ue4m3(block[4 + s]) * g;
                for (int i = 0; i < 16; i++)
                {
                    int code = (block[8 + 8 * s + (i & 7)] >> (i < 8 ? 0 : 4)) & 0xF;
                    float value = Fp4[code & 7] * ((code & 8) != 0 ? -1 : 1);
                    expected[b * 64 + 16 * s + i] = value * scale;
                }
            }
        }

        var actual = Decode(DataType.NVFP4_GS, bytes);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], 0f);
        }
    }

    /// <summary>The UE4M3 value of a byte (E4M3 without sign: bias 7, subnormals at exponent 0).</summary>
    private static float Ue4m3(int x)
    {
        int e = (x >> 3) & 15;
        int m = x & 7;
        return e == 0 ? m / 512f : (1 + m / 8f) * MathF.Pow(2, e - 7);
    }

    private static float WriteHalf(Span<byte> destination, float value)
    {
        var half = (Half)value;
        BinaryPrimitives.WriteHalfLittleEndian(destination, half);
        return (float)half;
    }

    private static float[] Decode(DataType type, byte[] bytes)
    {
        var format = Quant.FormatOf(type)!.Value;
        long count = bytes.Length / type.BlockBytes * type.BlockSize;
        var result = new float[count];
        fixed (byte* src = bytes)
        fixed (float* dst = result)
        {
            Quant.Dequantize(format, src, count, dst);
        }

        return result;
    }
}
