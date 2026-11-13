using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

public class ViewAndNormTests
{
    /// <summary>Qwen-style: reshape to heads, slice the last axis, RmsNorm with weight on the strided view.</summary>
    [Fact]
    public void SliceReshapeRmsNorm()
    {
        int T = 3;
        int H = 4;
        int D = 16;
        var x = Kit.Rand(T * H * 2 * D, 1);
        var w = Kit.Rand(D, 2);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, H * 2 * D);
        var wv = tg.WF("w", w, D);
        var xr = tg.B.Reshape(xin, 0, H, 2 * D);
        var q = tg.B.Slice(xr, 2, 0, D);
        var gate = tg.B.Slice(xr, 2, D, D);
        tg.B.Output("q", tg.B.RmsNorm(q, wv, 1e-6f));
        tg.B.Output("gate", tg.B.Mul(tg.B.Sigmoid(gate), tg.B.RmsNorm(gate, null, 1e-6f)));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(T);
        m.Run(p);
        var qs = new float[T * H * D];
        var gs = new float[T * H * D];
        for (int r = 0; r < T * H; r++)
        {
            for (int i = 0; i < D; i++)
            {
                qs[r * D + i] = x[r * 2 * D + i];
                gs[r * D + i] = x[r * 2 * D + D + i];
            }
        }

        Kit.AssertClose(Ref.RmsNorm(qs, T * H, D, w, 1e-6f), Kit.Out(m, "q", p), 1e-5, 1e-4);
        var gn = Ref.RmsNorm(gs, T * H, D, null, 1e-6f);
        Kit.AssertClose(gs.Select((v, i) => Ref.Sigmoid(v) * gn[i]).ToArray(), Kit.Out(m, "gate", p), 1e-5, 1e-4);
    }

    /// <summary>A strided slice reshaped into a shape that cannot alias (forces a materializing copy).</summary>
    [Fact]
    public void ReshapeOfStridedSliceMaterializes()
    {
        int T = 3;
        int H = 4;
        int D = 8;
        var x = Kit.Rand(T * H * 2 * D, 4);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, H * 2 * D);
        var q = tg.B.Slice(tg.B.Reshape(xin, 0, H, 2 * D), 2, 0, D);   // [T,H,D] strides (2HD, 2D, 1)
        var flat = tg.B.Reshape(q, 0, H * D);                           // not viewable
        tg.B.Output("y", tg.B.Scale(flat, 2f));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(T);
        m.Run(p);
        var exp = new float[T * H * D];
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int i = 0; i < D; i++)
                {
                    exp[(t * H + h) * D + i] = 2 * x[(t * H + h) * 2 * D + i];
                }
            }
        }

        Kit.AssertClose(exp, Kit.Out(m, "y", p));
    }

    [Fact]
    public void PermuteContiguousAndStridedConsumers()
    {
        int T = 3;
        int H = 5;
        int D = 7;
        var x = Kit.Rand(T * H * D, 9);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, H, D);
        var pm = tg.B.Permute(xin, 0, 2, 1);                          // [T, D, H]
        tg.B.Output("c", tg.B.Contiguous(pm));
        tg.B.Output("s", tg.B.Silu(pm));                               // last axis not contiguous => materialized
        tg.B.Output("sm", tg.B.Emit1("Softmax", [pm]));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(T);
        m.Run(p);
        var tr = new float[T * D * H];
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int d = 0; d < D; d++)
                {
                    tr[(t * D + d) * H + h] = x[(t * H + h) * D + d];
                }
            }
        }

        Kit.AssertClose(tr, Kit.Out(m, "c", p));
        Kit.AssertClose(tr.Select(Ref.Silu).ToArray(), Kit.Out(m, "s", p), 1e-5, 1e-5);
        Kit.AssertClose(Ref.Softmax(tr, T * D, H, 1f), Kit.Out(m, "sm", p), 1e-6, 1e-5);
    }

    [Fact]
    public void ConcatAlongAxes()
    {
        int T = 3;
        int A = 5;
        int B = 3;
        var a = Kit.Rand(T * A, 1);
        var b = Kit.Rand(T * B, 2);
        var tg = new TestGraph();
        var ain = tg.InF("a", TestGraph.T, A);
        var bin = tg.InF("b", TestGraph.T, 8);
        var bs = tg.B.Slice(bin, 1, 2, B); // strided
        tg.B.Output("y", tg.B.Concat(1, ain, bs, ain));
        using var m = Kit.Compile(tg.B.Build());
        var b8 = new float[T * 8];
        for (int t = 0; t < T; t++)
        {
            for (int i = 0; i < B; i++)
            {
                b8[t * 8 + 2 + i] = b[t * B + i];
            }
        }

        Kit.Set(m, ain, a);
        Kit.Set(m, bin, b8);
        var p = Kit.P(T);
        m.Run(p);
        var exp = new List<float>();
        for (int t = 0; t < T; t++)
        {
            exp.AddRange(a.AsSpan(t * A, A).ToArray());
            exp.AddRange(b.AsSpan(t * B, B).ToArray());
            exp.AddRange(a.AsSpan(t * A, A).ToArray());
        }
        Kit.AssertClose(exp.ToArray(), Kit.Out(m, "y", p));
    }

    [Fact]
    public void Norms()
    {
        int T = 5;
        int n = 70;
        var x = Kit.Rand(T * n, 3, 3f);
        var w = Kit.Rand(n, 4);
        var bias = Kit.Rand(n, 5);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, n);
        var wv = tg.WF("w", w, n);
        var bv = tg.WF("b", bias, n);
        tg.B.Output("rms", tg.B.RmsNorm(xin, wv, 1e-5f));
        tg.B.Output("rms0", tg.B.RmsNorm(xin, null, 1e-5f));
        tg.B.Output("l2", tg.B.L2Norm(xin, 1e-6f));
        tg.B.Output("ln", tg.B.Emit1("LayerNorm", [xin, wv, bv], Attributes.From(("eps", 1e-5f))));
        tg.B.Output("sm", tg.B.Emit1("Softmax", [xin], Attributes.From(("scale", 0.37f))));
        tg.B.Output("smneg", tg.B.Emit1("Softmax", [xin], Attributes.From(("scale", -0.5f))));
        tg.B.Output("sum", tg.B.Emit1("ReduceSum", [xin]));
        tg.B.Output("max", tg.B.Emit1("ReduceMax", [xin]));
        tg.B.Output("am", tg.B.Emit1("ArgMax", [xin]));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(T);
        m.Run(p);
        Kit.AssertClose(Ref.RmsNorm(x, T, n, w, 1e-5f), Kit.Out(m, "rms", p), 1e-5, 1e-4);
        Kit.AssertClose(Ref.RmsNorm(x, T, n, null, 1e-5f), Kit.Out(m, "rms0", p), 1e-5, 1e-4);
        Kit.AssertClose(Ref.L2Norm(x, T, n, 1e-6f), Kit.Out(m, "l2", p), 1e-6, 1e-4);
        var ln = new float[T * n];
        for (int t = 0; t < T; t++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++)
            {
                mean += x[t * n + i];
            }

            mean /= n;
            double v = 0;
            for (int i = 0; i < n; i++)
            {
                v += Math.Pow(x[t * n + i] - mean, 2);
            }

            v /= n;
            for (int i = 0; i < n; i++)
            {
                ln[t * n + i] = (float)((x[t * n + i] - mean) / Math.Sqrt(v + 1e-5) * w[i] + bias[i]);
            }
        }
        Kit.AssertClose(ln, Kit.Out(m, "ln", p), 1e-5, 1e-4);
        Kit.AssertClose(Ref.Softmax(x, T, n, 0.37f), Kit.Out(m, "sm", p), 1e-6, 1e-4);
        Kit.AssertClose(Ref.Softmax(x, T, n, -0.5f), Kit.Out(m, "smneg", p), 1e-6, 1e-4);
        Kit.AssertClose(Enumerable.Range(0, T).Select(t => (float)Enumerable.Range(0, n).Sum(i => (double)x[t * n + i])).ToArray(), Kit.Out(m, "sum", p), 1e-4, 1e-4);
        Kit.AssertClose(Enumerable.Range(0, T).Select(t => Enumerable.Range(0, n).Max(i => x[t * n + i])).ToArray(), Kit.Out(m, "max", p));
        var am = Kit.OutI(m, "am", p);
        for (int t = 0; t < T; t++)
        {
            Assert.Equal(Enumerable.Range(0, n).MaxBy(i => x[t * n + i]), am[t]);
        }
    }

    [Fact]
    public void ArgMaxTieReturnsFirst()
    {
        var x = new float[] { 1, 5, 5, 2, 7, 7, 7, 0 };
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, 4);
        tg.B.Output("am", tg.B.Emit1("ArgMax", [xin]));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(2);
        m.Run(p);
        Assert.Equal([1, 0], Kit.OutI(m, "am", p));
    }

    [Fact]
    public void CastRoundTrips()
    {
        var x = Kit.Rand(3 * 19, 5, 100f);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, 19);
        var h = tg.B.Emit1("Cast", [xin], Attributes.From(("type", "f16")));
        tg.B.Output("h", tg.B.Emit1("Cast", [h], Attributes.From(("type", "f32"))));
        var bf = tg.B.Emit1("Cast", [xin], Attributes.From(("type", "bf16")));
        tg.B.Output("bf", tg.B.Emit1("Cast", [bf], Attributes.From(("type", "f32"))));
        tg.B.Output("i", tg.B.Emit1("Cast", [xin], Attributes.From(("type", "i32"))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(3);
        m.Run(p);
        Kit.AssertClose(x.Select(v => (float)(Half)v).ToArray(), Kit.Out(m, "h", p), 0, 0);
        Kit.AssertClose(x.Select(v => { uint b = BitConverter.SingleToUInt32Bits(v); b += 0x7FFF + ((b >> 16) & 1); return BitConverter.UInt32BitsToSingle(b & 0xFFFF0000); }).ToArray(), Kit.Out(m, "bf", p), 0, 0);
        Assert.Equal(x.Select(v => (int)v).ToArray(), Kit.OutI(m, "i", p));
    }

    [Fact]
    public void GatherRowsOfF32Activation()
    {
        int T = 6;
        int D = 10;
        var x = Kit.Rand(T * D, 1);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, D);
        var rows = tg.In("rows", DataType.I32, Dim.Sym("R"));
        tg.B.Output("y", tg.B.Gather(tg.B.Scale(xin, 2f), rows));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        Kit.Set(m, rows, new[] { 5, 1 });
        var p = Kit.P(T, 0, ("R", 2));
        m.Run(p);
        var exp = new List<float>();
        foreach (int r in new[] { 5, 1 })
        {
            exp.AddRange(x.AsSpan(r * D, D).ToArray().Select(v => 2 * v));
        }

        Kit.AssertClose(exp.ToArray(), Kit.Out(m, "y", p));
    }
}
