using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// FP8 weight formats (OCP E4M3FN / E5M2 and the block-scaled f8e4m3_b32 / f8e4m3_b128) against hand-computed values and an
/// independent double-precision decoding of the bit fields, then MatMul / MatMulId / Gather against the dequantized product.
/// </summary>
public unsafe class Fp8DequantTests
{
    /// <summary>E4M3FN codes and their values (OCP 8-bit floating point specification, table of E4M3 encodings).</summary>
    [Theory]
    [InlineData(0x00, 0.0)]
    [InlineData(0x01, 0.001953125)]     // smallest subnormal 2^-9
    [InlineData(0x07, 0.013671875)]     // largest subnormal 7 * 2^-9
    [InlineData(0x08, 0.015625)]        // smallest normal 2^-6
    [InlineData(0x38, 1.0)]
    [InlineData(0x3F, 1.875)]
    [InlineData(0x40, 2.0)]
    [InlineData(0x4C, 6.0)]
    [InlineData(0x7E, 448.0)]           // largest finite
    [InlineData(0xB8, -1.0)]
    [InlineData(0xFE, -448.0)]
    [InlineData(0x81, -0.001953125)]
    public void E4m3KnownValues(int code, double expected)
    {
        Assert.Equal((float)expected, Decode(DataType.F8E4M3, [(byte)code])[0]);
    }

    [Theory]
    [InlineData(0x7F)]
    [InlineData(0xFF)]
    public void E4m3NanCodes(int code) => Assert.True(float.IsNaN(Decode(DataType.F8E4M3, [(byte)code])[0]));

    /// <summary>E5M2 codes: the upper byte of an IEEE half.</summary>
    [Theory]
    [InlineData(0x3C, 1.0)]
    [InlineData(0x01, 1.52587890625e-05)]   // 2^-16 (subnormal)
    [InlineData(0x7B, 57344.0)]             // largest finite
    [InlineData(0xC0, -2.0)]
    [InlineData(0x7C, double.PositiveInfinity)]
    public void E5m2KnownValues(int code, double expected)
    {
        Assert.Equal((float)expected, Decode(DataType.F8E5M2, [(byte)code])[0]);
    }

    [Fact]
    public void E4m3AllCodesMatchBitFieldFormula()
    {
        var codes = Enumerable.Range(0, 256).Select(c => (byte)c).ToArray();
        var values = Decode(DataType.F8E4M3, codes);
        for (int c = 0; c < 256; c++)
        {
            Assert.Equal(ReferenceE4m3(c), values[c]);
        }
    }

    [Theory]
    [InlineData("f8e4m3_b32")]
    [InlineData("f8e4m3_b128")]
    public void BlockScaledValuesAreScaleTimesCode(string typeName)
    {
        var type = DataType.Get(typeName);
        int blocks = 3;
        var (bytes, expected) = RandomBlocks(type, blocks, seed: 5);
        var values = Decode(type, bytes);
        Assert.Equal(expected, values);
    }

    [Theory]
    [InlineData("f8e4m3", 64, 7)]
    [InlineData("f8e5m2", 64, 7)]
    [InlineData("f8e4m3_b32", 96, 6)]
    [InlineData("f8e4m3_b128", 256, 5)]
    public void MatMulMatchesDequantizedProduct(string typeName, int cols, int rows)
    {
        var type = DataType.Get(typeName);
        var (bytes, dense) = RandomWeights(type, rows, cols, seed: 9);
        foreach (int tokens in new[] { 1, 4 })
        {
            var x = Kit.Rand(tokens * cols, 12);
            var tg = new TestGraph();
            var w = tg.W("w", type, new Shape(rows, cols), bytes);
            var xin = tg.InF("x", TestGraph.T, cols);
            tg.B.Output("y", tg.B.MatMul(xin, w));
            using var m = Kit.Compile(tg.B.Build());
            Kit.Set(m, xin, x);
            var p = Kit.P(tokens);
            m.Run(p);
            Kit.AssertClose(Product(x, dense, tokens, rows, cols), Kit.Out(m, "y", p), 1e-4, 1e-5, typeName + " matmul");
        }
    }

    [Theory]
    [InlineData("f8e4m3_b32", 64)]
    [InlineData("f8e4m3_b128", 128)]
    public void MatMulIdMatchesDequantizedProduct(string typeName, int cols)
    {
        var type = DataType.Get(typeName);
        const int experts = 3;
        const int outputs = 2;
        const int tokens = 3;
        int[] ids = [2, 0, 1, 1, 0, 2];   // [T, S] with S = 2 slots
        var (bytes, dense) = RandomWeights(type, experts * outputs, cols, seed: 4);
        var x = Kit.Rand(tokens * cols, 13);
        var tg = new TestGraph();
        var w = tg.W("w", type, new Shape(experts, outputs, cols), bytes);
        var xin = tg.InF("x", TestGraph.T, cols);
        var idin = tg.In("ids", DataType.I32, TestGraph.T, 2);
        tg.B.Output("y", tg.B.Emit1("MatMulId", [xin, w, idin]));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        Kit.Set(m, idin, ids);
        var p = Kit.P(tokens);
        m.Run(p);
        var expected = new float[tokens * 2 * outputs];
        for (int t = 0; t < tokens; t++)
        {
            for (int s = 0; s < 2; s++)
            {
                for (int n = 0; n < outputs; n++)
                {
                    double sum = 0;
                    int row = ids[t * 2 + s] * outputs + n;
                    for (int c = 0; c < cols; c++)
                    {
                        sum += (double)x[t * cols + c] * dense[row * cols + c];
                    }

                    expected[(t * 2 + s) * outputs + n] = (float)sum;
                }
            }
        }

        Kit.AssertClose(expected, Kit.Out(m, "y", p), 1e-4, 1e-5, typeName + " matmulid");
    }

    [Theory]
    [InlineData("f8e4m3", 32)]
    [InlineData("f8e4m3_b128", 128)]
    public void GatherMatchesDequantizedRows(string typeName, int cols)
    {
        var type = DataType.Get(typeName);
        const int rows = 5;
        var (bytes, dense) = RandomWeights(type, rows, cols, seed: 3);
        int[] idx = [4, 0, 2];
        var tg = new TestGraph();
        var table = tg.W("t", type, new Shape(rows, cols), bytes);
        var idin = tg.In("idx", DataType.I32, TestGraph.T);
        tg.B.Output("y", tg.B.Gather(table, idin));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, idin, idx);
        var p = Kit.P(idx.Length);
        m.Run(p);
        var expected = idx.SelectMany(r => dense.AsSpan(r * cols, cols).ToArray()).ToArray();
        Assert.Equal(expected, Kit.Out(m, "y", p));
    }

    /// <summary>Independent E4M3FN decoding in double precision from the sign, exponent and mantissa fields.</summary>
    private static float ReferenceE4m3(int code)
    {
        int sign = code >> 7;
        int exponent = (code >> 3) & 15;
        int mantissa = code & 7;
        if (exponent == 15 && mantissa == 7)
        {
            return float.NaN;
        }

        double magnitude = exponent == 0 ? mantissa / 8.0 * Math.Pow(2, -6) : (1 + mantissa / 8.0) * Math.Pow(2, exponent - 7);
        return (float)(sign == 1 ? -magnitude : magnitude);
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

    /// <summary>Random finite codes (no NaN / infinity) with f32 block scales, and their values decoded by the reference formula.</summary>
    private static (byte[] Bytes, float[] Values) RandomBlocks(DataType type, int blocks, int seed)
    {
        var rng = new Random(seed);
        int block = type.BlockSize;
        var bytes = new byte[blocks * type.BlockBytes];
        var values = new float[blocks * block];
        for (int b = 0; b < blocks; b++)
        {
            float d = (float)(rng.NextDouble() * 0.01 + 1e-4);
            BitConverter.TryWriteBytes(bytes.AsSpan(b * type.BlockBytes, 4), d);
            for (int i = 0; i < block; i++)
            {
                byte code = FiniteCode(rng, e5m2: false);
                bytes[b * type.BlockBytes + 4 + i] = code;
                values[b * block + i] = ReferenceE4m3(code) * d;
            }
        }

        return (bytes, values);
    }

    /// <summary>Random weights of any FP8 type as storage bytes and dense values.</summary>
    private static (byte[] Bytes, float[] Values) RandomWeights(DataType type, int rows, int cols, int seed)
    {
        if (type.BlockSize > 1)
        {
            return RandomBlocks(type, rows * cols / type.BlockSize, seed);
        }

        var rng = new Random(seed);
        bool e5m2 = type == DataType.F8E5M2;
        var bytes = new byte[rows * cols];
        var values = new float[rows * cols];
        for (int i = 0; i < bytes.Length; i++)
        {
            bytes[i] = FiniteCode(rng, e5m2);
            values[i] = e5m2 ? (float)BitConverter.UInt16BitsToHalf((ushort)(bytes[i] << 8)) : ReferenceE4m3(bytes[i]);
        }

        return (bytes, values);
    }

    /// <summary>A random code below magnitude 2^4 (E5M2) or the full finite range (E4M3).</summary>
    private static byte FiniteCode(Random rng, bool e5m2)
    {
        while (true)
        {
            byte code = (byte)rng.Next(256);
            bool finite = e5m2 ? ((code >> 2) & 31) < 19 : (code & 0x7F) != 0x7F;
            if (finite)
            {
                return code;
            }
        }
    }

    private static float[] Product(float[] x, float[] dense, int tokens, int rows, int cols)
    {
        var expected = new float[tokens * rows];
        for (int t = 0; t < tokens; t++)
        {
            for (int r = 0; r < rows; r++)
            {
                double sum = 0;
                for (int c = 0; c < cols; c++)
                {
                    sum += (double)x[t * cols + c] * dense[r * cols + c];
                }

                expected[t * rows + r] = (float)sum;
            }
        }

        return expected;
    }
}
