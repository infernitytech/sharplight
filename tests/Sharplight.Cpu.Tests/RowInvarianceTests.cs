using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// Row invariance of the CPU matmuls: a step of 2-4 rows (a speculative verification step) must compute every row bit-identically to a
/// step of that row alone, over the weight formats and the small K / N of tiny and hyper-connection projections.
/// </summary>
public sealed class RowInvarianceTests
{
    [Theory]
    [InlineData("bf16", 4, 32)]
    [InlineData("bf16", 8, 640)]
    [InlineData("bf16", 32, 4)]
    [InlineData("bf16", 384, 8)]
    [InlineData("f32", 8, 384)]
    [InlineData("f16", 64, 96)]
    [InlineData("q8_0", 256, 96)]
    public void MatMul(string fmt, int k, int n)
    {
        const int Rows = 4;
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, k);
        var type = DataType.Get(fmt);
        var w = tg.W("w", type, new Shape(n, k), Quantized(type, Kit.Rand(n * k, 5, 0.1f), n, k), zeroCopy: false);
        tg.B.Output("y", tg.B.MatMul(x, w));
        using var m = Kit.Compile(tg.B.Build(), intDot: true);
        float[] data = Kit.Rand(Rows * k, 9);
        var single = new List<float[]>();
        for (int r = 0; r < Rows; r++)
        {
            Kit.Set(m, x, data[(r * k)..((r + 1) * k)]);
            var p = Kit.P(1);
            m.Run(p);
            single.Add(Kit.Out(m, "y", p));
        }

        for (int t = 2; t <= Rows; t++)
        {
            Kit.Set(m, x, data[..(t * k)]);
            var p = Kit.P(t);
            m.Run(p);
            var batched = Kit.Out(m, "y", p);
            for (int r = 0; r < t; r++)
            {
                for (int i = 0; i < n; i++)
                {
                    if (BitConverter.SingleToInt32Bits(batched[r * n + i]) != BitConverter.SingleToInt32Bits(single[r][i]))
                    {
                        Assert.Fail($"{t}-row step, row {r}, column {i}: {batched[r * n + i]:R} vs {single[r][i]:R} alone");
                    }
                }
            }
        }
    }

    private static unsafe byte[] Quantized(DataType type, float[] values, int n, int k)
    {
        if (type == DataType.F32)
        {
            return System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
        }

        if (type == DataType.BF16 || type == DataType.F16)
        {
            var bytes = new byte[values.Length * 2];
            for (int i = 0; i < values.Length; i++)
            {
                ushort bits = type == DataType.BF16 ? (ushort)(BitConverter.SingleToUInt32Bits(values[i]) >> 16) : BitConverter.HalfToUInt16Bits((Half)values[i]);
                BitConverter.TryWriteBytes(bytes.AsSpan(i * 2), bits);
            }

            return bytes;
        }

        var quantized = new byte[type.ByteSize(k) * n];
        fixed (float* src = values)
        {
            fixed (byte* dst = quantized)
            {
                Quant.Quantize(WeightFormat.Q8_0, src, values.Length, dst);
            }
        }

        return quantized;
    }
}
