using System.Runtime.InteropServices;
using Sharplight.Cpu;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// Group-wise 1-bit / ternary formats (Q1_0, Q2_0, Q2_0_G128, Tern1_G128). The reference decoders below follow the
/// published layouts element by element (independent of the optimized LUT implementation).
/// </summary>
public unsafe class BonsaiQuantTests
{
    static readonly DataType[] Types = [DataType.Q1_0, DataType.Q2_0, DataType.Q2_0_G128, DataType.Tern1_G128];

    public static IEnumerable<object[]> TypeNames() => Types.Select(t => new object[] { t.Name });

    static byte[] RandomBlocks(DataType t, long elements, int seed)
    {
        var rng = new Random(seed);
        var b = new byte[t.ByteSize(elements)];
        rng.NextBytes(b);
        long nb = elements / t.BlockSize;
        for (long i = 0; i < nb; i++)
        {
            long o = i * t.BlockBytes + (t == DataType.Tern1_G128 ? 26 : 0);
            BitConverter.TryWriteBytes(b.AsSpan((int)o, 2), (Half)(0.01f + (float)rng.NextDouble() * 0.05f));
        }
        return b;
    }

    static int Trit(byte b, int n) { int pw = 1; for (int i = 0; i < n; i++) pw *= 3; return ((((b * pw) & 0xFF) * 3) >> 8) - 1; }

    public static float[] Reference(DataType t, byte[] q, long elements)
    {
        var r = new float[elements];
        long nb = elements / t.BlockSize;
        for (long blk = 0; blk < nb; blk++)
        {
            int o = (int)(blk * t.BlockBytes);
            long e0 = blk * t.BlockSize;
            if (t == DataType.Q1_0)
            {
                float d = (float)BitConverter.ToHalf(q, o);
                for (int j = 0; j < 128; j++) r[e0 + j] = ((q[o + 2 + j / 8] >> (j % 8)) & 1) != 0 ? d : -d;
            }
            else if (t == DataType.Q2_0 || t == DataType.Q2_0_G128)
            {
                float d = (float)BitConverter.ToHalf(q, o);
                for (int j = 0; j < t.BlockSize; j++) r[e0 + j] = (((q[o + 2 + j / 4] >> (2 * (j % 4))) & 3) - 1) * d;
            }
            else
            {
                float d = (float)BitConverter.ToHalf(q, o + 26);
                int j = 0;
                // stage 1: 16 bytes x 5 trits, n-major; stage 2: 8 bytes x 5 trits; qh: 2 bytes x 4 trits
                for (int n = 0; n < 5; n++) for (int m = 0; m < 16; m++) r[e0 + j++] = Trit(q[o + m], n) * d;
                for (int n = 0; n < 5; n++) for (int m = 0; m < 8; m++) r[e0 + j++] = Trit(q[o + 16 + m], n) * d;
                for (int n = 0; n < 4; n++) for (int h = 0; h < 2; h++) r[e0 + j++] = Trit(q[o + 24 + h], n) * d;
            }
        }
        return r;
    }

    [Theory, MemberData(nameof(TypeNames))]
    public void DequantizeMatchesReference(string name)
    {
        var t = DataType.Get(name);
        long n = t.BlockSize * 13;
        var q = RandomBlocks(t, n, 5);
        var exp = Reference(t, q, n);
        var act = new float[n];
        fixed (byte* p = q) fixed (float* d = act) Quant.Dequantize(Quant.FormatOf(t)!.Value, p, n, d);
        Kit.AssertClose(exp, act, 0, 0, name);
    }

    [Theory, MemberData(nameof(TypeNames))]
    public void GatherAndMatMul(string name)
    {
        var t = DataType.Get(name);
        int rows = 9, k = t.BlockSize * 4;
        var q = RandomBlocks(t, (long)rows * k, 9);
        var deq = Reference(t, q, (long)rows * k);

        var tg = new TG();
        var table = tg.W("tbl", t, new Shape(rows, k), q);
        var idx = tg.In("idx", DataType.I32, TG.T);
        tg.B.Output("y", tg.B.Gather(table, idx));
        using (var m = Kit.Compile(tg.B.Build()))
        {
            m.SetInput(idx, MemoryMarshal.AsBytes(new[] { 3, 0, 8 }.AsSpan()));
            var p = Kit.P(3); m.Run(p);
            var exp = new List<float>();
            foreach (int r in new[] { 3, 0, 8 }) exp.AddRange(deq.AsSpan(r * k, k).ToArray());
            Kit.AssertClose(exp.ToArray(), Kit.Out(m, "y", p), 0, 0, name + " gather");
        }

        foreach (int tokens in new[] { 1, 4 })
        {
            var x = Kit.Rand(tokens * k, 11);
            var tg2 = new TG();
            var w = tg2.W("w", t, new Shape(rows, k), q);
            var xin = tg2.InF("x", TG.T, k);
            tg2.B.Output("y", tg2.B.MatMul(xin, w));
            using var m = Kit.Compile(tg2.B.Build());
            Kit.Set(m, xin, x);
            var p = Kit.P(tokens); m.Run(p);
            var exp = new float[tokens * rows];
            for (int tt = 0; tt < tokens; tt++)
                for (int r = 0; r < rows; r++)
                {
                    double s = 0; for (int c = 0; c < k; c++) s += (double)x[tt * k + c] * deq[r * k + c];
                    exp[tt * rows + r] = (float)s;
                }
            Kit.AssertClose(exp, Kit.Out(m, "y", p), 2e-3, 1e-4, name + " matmul");
        }
    }
}
