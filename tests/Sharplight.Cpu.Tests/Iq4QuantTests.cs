using System.Runtime.InteropServices;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>ggml IQ4_NL / IQ4_XS (non-linear 4-bit codebook formats); the reference decoders follow the published block layouts element by element.</summary>
public unsafe class Iq4QuantTests
{
    private static readonly int[] KValues = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
    private static readonly DataType[] Types = [DataType.IQ4_NL, DataType.IQ4_XS];

    public static IEnumerable<object[]> TypeNames() => Types.Select(t => new object[] { t.Name });

    private static byte[] RandomBlocks(DataType t, long elements, int seed)
    {
        var rng = new Random(seed);
        var b = new byte[t.ByteSize(elements)];
        rng.NextBytes(b);
        for (long i = 0; i < elements / t.BlockSize; i++)
        {
            BitConverter.TryWriteBytes(b.AsSpan((int)(i * t.BlockBytes), 2), (Half)(0.002f + (float)rng.NextDouble() * 0.01f));
        }

        return b;
    }

    public static float[] Reference(DataType t, byte[] q, long elements)
    {
        var r = new float[elements];
        for (long blk = 0; blk < elements / t.BlockSize; blk++)
        {
            int o = (int)(blk * t.BlockBytes);
            float d = (float)BitConverter.ToHalf(q, o);
            long e0 = blk * t.BlockSize;
            if (t == DataType.IQ4_NL)
            {
                for (int j = 0; j < 16; j++)
                {
                    r[e0 + j] = d * KValues[q[o + 2 + j] & 15];
                    r[e0 + j + 16] = d * KValues[q[o + 2 + j] >> 4];
                }
                continue;
            }
            int scalesH = q[o + 2] | q[o + 3] << 8;
            for (int ib = 0; ib < 8; ib++)
            {
                int ls = ((q[o + 4 + ib / 2] >> (4 * (ib % 2))) & 15) | (((scalesH >> (2 * ib)) & 3) << 4);
                float dl = d * (ls - 32);
                for (int j = 0; j < 16; j++)
                {
                    byte b = q[o + 8 + 16 * ib + j];
                    r[e0 + ib * 32 + j] = dl * KValues[b & 15];
                    r[e0 + ib * 32 + j + 16] = dl * KValues[b >> 4];
                }
            }
        }
        return r;
    }

    [Theory, MemberData(nameof(TypeNames))]
    public void DequantizeMatchesReference(string name)
    {
        var t = DataType.Get(name);
        long n = t.BlockSize * 11;
        var q = RandomBlocks(t, n, 3);
        var act = new float[n];
        fixed (byte* p = q) fixed (float* d = act)
        {
            Quant.Dequantize(Quant.FormatOf(t)!.Value, p, n, d);
        }

        Kit.AssertClose(Reference(t, q, n), act, 0, 0, name);
    }

    [Theory, MemberData(nameof(TypeNames))]
    public void GatherAndMatMul(string name)
    {
        var t = DataType.Get(name);
        int rows = 7;
        int k = t.BlockSize * 3;
        var q = RandomBlocks(t, (long)rows * k, 4);
        var deq = Reference(t, q, (long)rows * k);
        var tg = new TestGraph();
        var table = tg.W("tbl", t, new Shape(rows, k), q);
        var idx = tg.In("idx", DataType.I32, TestGraph.T);
        tg.B.Output("y", tg.B.Gather(table, idx));
        using (var m = Kit.Compile(tg.B.Build()))
        {
            m.SetInput(idx, MemoryMarshal.AsBytes(new[] { 6, 0, 2 }.AsSpan()));
            var p = Kit.P(3);
            m.Run(p);
            var exp = new List<float>();
            foreach (int r in new[] { 6, 0, 2 })
            {
                exp.AddRange(deq.AsSpan(r * k, k).ToArray());
            }

            Kit.AssertClose(exp.ToArray(), Kit.Out(m, "y", p), 0, 0, name + " gather");
        }
        foreach (int tokens in new[] { 1, 5 })
        {
            var x = Kit.Rand(tokens * k, 12);
            var tg2 = new TestGraph();
            var w = tg2.W("w", t, new Shape(rows, k), q);
            var xin = tg2.InF("x", TestGraph.T, k);
            tg2.B.Output("y", tg2.B.MatMul(xin, w));
            using var m = Kit.Compile(tg2.B.Build());
            Kit.Set(m, xin, x);
            var p = Kit.P(tokens);
            m.Run(p);
            var exp = new float[tokens * rows];
            for (int tt = 0; tt < tokens; tt++)
            {
                for (int r = 0; r < rows; r++)
                {
                    double s = 0;
                    for (int c = 0; c < k; c++)
                    {
                        s += (double)x[tt * k + c] * deq[r * k + c];
                    }

                    exp[tt * rows + r] = (float)s;
                }
            }

            Kit.AssertClose(exp, Kit.Out(m, "y", p), 2e-3, 1e-4, name + " matmul");
        }
    }
}
