using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

public class ElementwiseTests
{
    private static readonly Dictionary<string, Func<double, double>> UnaryRef = new()
    {
        ["Neg"] = x => -x, ["Exp"] = Math.Exp, ["Log"] = Math.Log, ["Sqrt"] = Math.Sqrt, ["Rsqrt"] = x => 1 / Math.Sqrt(x),
        ["Sigmoid"] = x => 1 / (1 + Math.Exp(-x)), ["Silu"] = x => x / (1 + Math.Exp(-x)),
        ["Gelu"] = x => 0.5 * x * (1 + Math2.Erf(x / Math.Sqrt(2))),
        ["GeluTanh"] = x => 0.5 * x * (1 + Math.Tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))),
        ["Tanh"] = Math.Tanh, ["Softplus"] = x => x > 20 ? x : Math.Log(1 + Math.Exp(x)), ["Relu"] = x => Math.Max(0, x),
        ["Abs"] = Math.Abs, ["Square"] = x => x * x, ["Recip"] = x => 1 / x,
        ["Erf"] = Math2.Erf, ["HardSwish"] = x => x * Math.Clamp(x / 6 + 0.5, 0, 1), ["HardSigmoid"] = x => Math.Clamp(0.2 * x + 0.5, 0, 1),
    };

    public static IEnumerable<object[]> UnaryNames() => StandardOps.UnaryNames.Where(n => UnaryRef.ContainsKey(n)).Select(n => new object[] { n });

    [Theory, MemberData(nameof(UnaryNames))]
    public void Unary(string op)
    {
        bool positive = op is "Log" or "Sqrt" or "Rsqrt" or "Recip";
        int T = 3, n = 37;
        var x = Kit.Rand(T * n, 1, op == "Softplus" ? 30f : 6f);
        if (positive)
        {
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = Math.Abs(x[i]) + 0.1f;
            }
        }

        var tg = new TG(); var xin = tg.InF("x", TG.T, n);
        tg.B.Output("y", tg.B.Unary(op, xin));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(T); m.Run(p);
        Kit.AssertClose(x.Select(v => (float)UnaryRef[op](v)).ToArray(), Kit.Out(m, "y", p), 1e-5, 2e-5, op);
    }

    [Fact]
    public void LargeUnaryUsesParallelPath()
    {
        int T = 64, n = 2048; var x = Kit.Rand(T * n, 2, 6f);
        var tg = new TG(); var xin = tg.InF("x", TG.T, n);
        tg.B.Output("y", tg.B.Silu(xin));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(T); m.Run(p);
        Kit.AssertClose(x.Select(v => Ref.Silu(v)).ToArray(), Kit.Out(m, "y", p), 1e-5, 2e-5);
    }

    [Fact]
    public void ErfAccuracy()
    {
        (double x, double e)[] t = [(0.1, 0.1124629160182849), (0.5, 0.5204998778130465), (1, 0.8427007929497149), (-1.5, -0.9661051464753108),
            (2, 0.9953222650189527), (2.5, 0.9995930479825550), (3, 0.9999779095030014), (4, 0.9999999845827421), (0, 0)];
        foreach (var (x, e) in t)
        {
            Assert.True(Math.Abs(Math2.Erf(x) - e) < 1e-12, $"erf({x}) = {Math2.Erf(x)} vs {e}");
        }
    }

    [Fact]
    public void ScaleOp()
    {
        var x = Kit.Rand(3 * 21, 3);
        var tg = new TG(); var xin = tg.InF("x", TG.T, 21);
        tg.B.Output("y", tg.B.Scale(xin, 2.5f, -0.25f));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(3); m.Run(p);
        Kit.AssertClose(x.Select(v => v * 2.5f - 0.25f).ToArray(), Kit.Out(m, "y", p));
    }

    private static float[] BcastRef(string op, float[] a, long[] ad, float[] b, long[] bd, out long[] od)
    {
        int r = Math.Max(ad.Length, bd.Length);
        od = new long[r];
        long[] pa = Enumerable.Repeat(1L, r - ad.Length).Concat(ad).ToArray(), pb = Enumerable.Repeat(1L, r - bd.Length).Concat(bd).ToArray();
        for (int i = 0; i < r; i++)
        {
            od[i] = Math.Max(pa[i], pb[i]);
        }

        long total = od.Aggregate(1L, (p, d) => p * d);
        var y = new float[total];
        for (long lin = 0; lin < total; lin++)
        {
            long rem = lin, ia = 0, ib = 0, sa = 1, sb = 1;
            for (int d = r - 1; d >= 0; d--)
            {
                long idx = rem % od[d]; rem /= od[d];
                ia += (pa[d] == 1 ? 0 : idx) * sa; ib += (pb[d] == 1 ? 0 : idx) * sb; sa *= pa[d]; sb *= pb[d];
            }
            float u = a[ia], v = b[ib];
            y[lin] = op switch { "Add" => u + v, "Sub" => u - v, "Mul" => u * v, "Div" => u / v, "Max" => Math.Max(u, v), "Pow" => (float)Math.Pow(u, v), _ => Math.Min(u, v) };
        }
        return y;
    }

    public static IEnumerable<object[]> BinaryCases()
    {
        foreach (var op in StandardOps.BinaryNames.Where(o => o is "Add" or "Sub" or "Mul" or "Div" or "Max" or "Min" or "Pow"))
        {
            foreach (var (name, ad, bd) in new (string, long[], long[])[]
            {
                ("rowvec", [0, 48], [48]), ("colvec", [0, 6], [0, 1]), ("head", [0, 4, 8], [0, 4, 1]),
                ("same", [0, 37], [0, 37]), ("lastdim", [0, 4, 8], [8]), ("scalar", [0, 5], [1]),
            })
            {
                yield return [op, name];
            }
        }
    }

    [Theory, MemberData(nameof(BinaryCases))]
    public void BinaryBroadcast(string op, string name)
    {
        long T = 5;
        long[] ad = name switch { "rowvec" => [T, 48], "colvec" => [T, 6], "head" => [T, 4, 8], "same" => [T, 37], "lastdim" => [T, 4, 8], _ => [T, 5] };
        long[] bd = name switch { "rowvec" => [48], "colvec" => [T, 1], "head" => [T, 4, 1], "same" => [T, 37], "lastdim" => [8], _ => [1] };
        long an = ad.Aggregate(1L, (p, d) => p * d), bn = bd.Aggregate(1L, (p, d) => p * d);
        var a = Kit.Rand((int)an, 5); var b = Kit.Rand((int)bn, 6);
        if (op == "Div")
        {
            for (int i = 0; i < b.Length; i++)
            {
                b[i] = Math.Abs(b[i]) + 0.5f;
            }
        }

        if (op == "Pow")
        {
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = Math.Abs(a[i]) + 0.1f;
            }
        }

        var tg = new TG();
        var ain = tg.InF("a", [TG.T, .. ad.Skip(1).Select(d => (Dim)d)]);
        Value bv;
        Value? bin = null;
        if (bd[0] == T) { bin = tg.InF("b", [TG.T, .. bd.Skip(1).Select(d => (Dim)d)]); bv = bin; }
        else
        {
            bv = tg.WF("b", b, bd);
        }

        tg.B.Output("y", tg.B.Binary(op, ain, bv));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, ain, a); if (bin != null)
        {
            Kit.Set(m, bin, b);
        }

        var p = Kit.P(T); m.Run(p);
        Kit.AssertClose(BcastRef(op, a, ad, b, bd, out _), Kit.Out(m, "y", p), 1e-6, 1e-6, $"{op} {name}");
    }

    [Fact]
    public void BinaryOnStridedViewAndWeightInF16()
    {
        // y = slice(x)[:, 8:40] * w16[32]
        int T = 4, wide = 64, n = 32;
        var x = Kit.Rand(T * wide, 7);
        var wf = Kit.Rand(n, 8);
        var w16 = new byte[n * 2]; var wr = new float[n];
        for (int i = 0; i < n; i++) { var h = (Half)wf[i]; BitConverter.TryWriteBytes(w16.AsSpan(i * 2), h); wr[i] = (float)h; }
        var tg = new TG(); var xin = tg.InF("x", TG.T, wide);
        var w = tg.W("w", DataType.F16, new Shape(n), w16);
        tg.B.Output("y", tg.B.Mul(tg.B.Slice(xin, 1, 8, n), w));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(T); m.Run(p);
        var exp = new float[T * n]; for (int t = 0; t < T; t++)
        {
            for (int i = 0; i < n; i++)
            {
                exp[t * n + i] = x[t * wide + 8 + i] * wr[i];
            }
        }

        Kit.AssertClose(exp, Kit.Out(m, "y", p));
    }
}

public class ViewAndNormTests
{
    /// <summary>Qwen-style: reshape to heads, slice the last axis, RmsNorm with weight on the strided view.</summary>
    [Fact]
    public void SliceReshapeRmsNorm()
    {
        int T = 3, H = 4, D = 16;
        var x = Kit.Rand(T * H * 2 * D, 1); var w = Kit.Rand(D, 2);
        var tg = new TG(); var xin = tg.InF("x", TG.T, H * 2 * D); var wv = tg.WF("w", w, D);
        var xr = tg.B.Reshape(xin, 0, H, 2 * D);
        var q = tg.B.Slice(xr, 2, 0, D); var gate = tg.B.Slice(xr, 2, D, D);
        tg.B.Output("q", tg.B.RmsNorm(q, wv, 1e-6f));
        tg.B.Output("gate", tg.B.Mul(tg.B.Sigmoid(gate), tg.B.RmsNorm(gate, null, 1e-6f)));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(T); m.Run(p);
        var qs = new float[T * H * D]; var gs = new float[T * H * D];
        for (int r = 0; r < T * H; r++)
        {
            for (int i = 0; i < D; i++) { qs[r * D + i] = x[r * 2 * D + i]; gs[r * D + i] = x[r * 2 * D + D + i]; }
        }

        Kit.AssertClose(Ref.RmsNorm(qs, T * H, D, w, 1e-6f), Kit.Out(m, "q", p), 1e-5, 1e-4);
        var gn = Ref.RmsNorm(gs, T * H, D, null, 1e-6f);
        Kit.AssertClose(gs.Select((v, i) => Ref.Sigmoid(v) * gn[i]).ToArray(), Kit.Out(m, "gate", p), 1e-5, 1e-4);
    }

    /// <summary>A strided slice reshaped into a shape that cannot alias (forces a materializing copy).</summary>
    [Fact]
    public void ReshapeOfStridedSliceMaterializes()
    {
        int T = 3, H = 4, D = 8;
        var x = Kit.Rand(T * H * 2 * D, 4);
        var tg = new TG(); var xin = tg.InF("x", TG.T, H * 2 * D);
        var q = tg.B.Slice(tg.B.Reshape(xin, 0, H, 2 * D), 2, 0, D);   // [T,H,D] strides (2HD, 2D, 1)
        var flat = tg.B.Reshape(q, 0, H * D);                           // not viewable
        tg.B.Output("y", tg.B.Scale(flat, 2f));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(T); m.Run(p);
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
        int T = 3, H = 5, D = 7;
        var x = Kit.Rand(T * H * D, 9);
        var tg = new TG(); var xin = tg.InF("x", TG.T, H, D);
        var pm = tg.B.Permute(xin, 0, 2, 1);                          // [T, D, H]
        tg.B.Output("c", tg.B.Contiguous(pm));
        tg.B.Output("s", tg.B.Silu(pm));                               // last axis not contiguous => materialized
        tg.B.Output("sm", tg.B.Emit1("Softmax", [pm]));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(T); m.Run(p);
        var tr = new float[T * D * H]; for (int t = 0; t < T; t++)
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
        int T = 3, A = 5, B = 3;
        var a = Kit.Rand(T * A, 1); var b = Kit.Rand(T * B, 2);
        var tg = new TG(); var ain = tg.InF("a", TG.T, A); var bin = tg.InF("b", TG.T, 8);
        var bs = tg.B.Slice(bin, 1, 2, B); // strided
        tg.B.Output("y", tg.B.Concat(1, ain, bs, ain));
        using var m = Kit.Compile(tg.B.Build());
        var b8 = new float[T * 8]; for (int t = 0; t < T; t++)
        {
            for (int i = 0; i < B; i++)
            {
                b8[t * 8 + 2 + i] = b[t * B + i];
            }
        }

        Kit.Set(m, ain, a); Kit.Set(m, bin, b8);
        var p = Kit.P(T); m.Run(p);
        var exp = new List<float>(); for (int t = 0; t < T; t++) { exp.AddRange(a.AsSpan(t * A, A).ToArray()); exp.AddRange(b.AsSpan(t * B, B).ToArray()); exp.AddRange(a.AsSpan(t * A, A).ToArray()); }
        Kit.AssertClose(exp.ToArray(), Kit.Out(m, "y", p));
    }

    [Fact]
    public void Norms()
    {
        int T = 5, n = 70; var x = Kit.Rand(T * n, 3, 3f); var w = Kit.Rand(n, 4); var bias = Kit.Rand(n, 5);
        var tg = new TG(); var xin = tg.InF("x", TG.T, n); var wv = tg.WF("w", w, n); var bv = tg.WF("b", bias, n);
        tg.B.Output("rms", tg.B.RmsNorm(xin, wv, 1e-5f)); tg.B.Output("rms0", tg.B.RmsNorm(xin, null, 1e-5f));
        tg.B.Output("l2", tg.B.L2Norm(xin, 1e-6f));
        tg.B.Output("ln", tg.B.Emit1("LayerNorm", [xin, wv, bv], Attributes.From(("eps", 1e-5f))));
        tg.B.Output("sm", tg.B.Emit1("Softmax", [xin], Attributes.From(("scale", 0.37f))));
        tg.B.Output("smneg", tg.B.Emit1("Softmax", [xin], Attributes.From(("scale", -0.5f))));
        tg.B.Output("sum", tg.B.Emit1("ReduceSum", [xin])); tg.B.Output("max", tg.B.Emit1("ReduceMax", [xin])); tg.B.Output("am", tg.B.Emit1("ArgMax", [xin]));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(T); m.Run(p);
        Kit.AssertClose(Ref.RmsNorm(x, T, n, w, 1e-5f), Kit.Out(m, "rms", p), 1e-5, 1e-4);
        Kit.AssertClose(Ref.RmsNorm(x, T, n, null, 1e-5f), Kit.Out(m, "rms0", p), 1e-5, 1e-4);
        Kit.AssertClose(Ref.L2Norm(x, T, n, 1e-6f), Kit.Out(m, "l2", p), 1e-6, 1e-4);
        var ln = new float[T * n];
        for (int t = 0; t < T; t++)
        {
            double mean = 0; for (int i = 0; i < n; i++)
            {
                mean += x[t * n + i];
            }

            mean /= n;
            double v = 0; for (int i = 0; i < n; i++)
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
        var tg = new TG(); var xin = tg.InF("x", TG.T, 4);
        tg.B.Output("am", tg.B.Emit1("ArgMax", [xin]));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(2); m.Run(p);
        Assert.Equal([1, 0], Kit.OutI(m, "am", p));
    }

    [Fact]
    public void CastRoundTrips()
    {
        var x = Kit.Rand(3 * 19, 5, 100f);
        var tg = new TG(); var xin = tg.InF("x", TG.T, 19);
        var h = tg.B.Emit1("Cast", [xin], Attributes.From(("type", "f16")));
        tg.B.Output("h", tg.B.Emit1("Cast", [h], Attributes.From(("type", "f32"))));
        var bf = tg.B.Emit1("Cast", [xin], Attributes.From(("type", "bf16")));
        tg.B.Output("bf", tg.B.Emit1("Cast", [bf], Attributes.From(("type", "f32"))));
        tg.B.Output("i", tg.B.Emit1("Cast", [xin], Attributes.From(("type", "i32"))));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(3); m.Run(p);
        Kit.AssertClose(x.Select(v => (float)(Half)v).ToArray(), Kit.Out(m, "h", p), 0, 0);
        Kit.AssertClose(x.Select(v => { uint b = BitConverter.SingleToUInt32Bits(v); b += 0x7FFF + ((b >> 16) & 1); return BitConverter.UInt32BitsToSingle(b & 0xFFFF0000); }).ToArray(), Kit.Out(m, "bf", p), 0, 0);
        Assert.Equal(x.Select(v => (int)v).ToArray(), Kit.OutI(m, "i", p));
    }

    [Fact]
    public void GatherRowsOfF32Activation()
    {
        int T = 6, D = 10; var x = Kit.Rand(T * D, 1);
        var tg = new TG(); var xin = tg.InF("x", TG.T, D); var rows = tg.In("rows", DataType.I32, Dim.Sym("R"));
        tg.B.Output("y", tg.B.Gather(tg.B.Scale(xin, 2f), rows));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x); Kit.Set(m, rows, new[] { 5, 1 });
        var p = Kit.P(T, 0, ("R", 2)); m.Run(p);
        var exp = new List<float>(); foreach (int r in new[] { 5, 1 })
        {
            exp.AddRange(x.AsSpan(r * D, D).ToArray().Select(v => 2 * v));
        }

        Kit.AssertClose(exp.ToArray(), Kit.Out(m, "y", p));
    }
}

public class RopeTests
{
    public static IEnumerable<object[]> Cases() =>
    [
        ["neox", 16, 1, null!, false, 1f],
        ["gptj", 16, 1, null!, false, 1f],
        ["neox", 8, 1, null!, false, 0.5f],                // partial rotation, freq_scale
        ["gptj", 8, 1, null!, false, 1f],
        ["neox", 16, 4, new[] { 2, 3, 3, 0 }, false, 1f],  // M-RoPE contiguous sections
        ["neox", 16, 4, new[] { 3, 2, 2, 1 }, true, 1f],   // IMROPE
        ["neox", 16, 4, new[] { 4, 2, 2, 0 }, true, 1f],   // IMROPE, axis 3 absent
        ["neox", 8, 4, new[] { 1, 1, 1, 1 }, true, 1f],
    ];

    [Theory, MemberData(nameof(Cases))]
    public void RopeMatchesReference(string style, int dims, int axes, int[]? sections, bool interleaved, float freqScale)
    {
        int T = 6, H = 3, D = 16; float @base = 10000f;
        var x = Kit.Rand(T * H * D, 1);
        var rng = new Random(5); var pos = new int[T * axes];
        for (int t = 0; t < T; t++)
        {
            for (int a = 0; a < axes; a++)
            {
                pos[t * axes + a] = rng.Next(0, 300);
            }
        }

        var tg = new TG(); var xin = tg.InF("x", TG.T, H, D); var pin = tg.In("pos", DataType.I32, TG.T, axes);
        var attrs = Attributes.From(("dims", dims), ("base", @base), ("style", style), ("interleaved", interleaved), ("freq_scale", freqScale))
            .With("sections", sections ?? [0, 0, 0, 0]);
        tg.B.Output("y", tg.B.Emit1("Rope", [xin, pin], attrs));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x); Kit.Set(m, pin, pos);
        var p = Kit.P(T); m.Run(p);
        Kit.AssertClose(Ref.Rope(x, T, H, D, pos, axes, dims, @base, style, sections, interleaved, freqScale), Kit.Out(m, "y", p), 1e-4, 1e-4);
    }

    /// <summary>Per-frequency divisors as third input (llama3 / yarn scaling).</summary>
    [Fact]
    public void RopeWithFrequencyFactors()
    {
        int T = 5, H = 2, D = 16; float @base = 500000f;
        var x = Kit.Rand(T * H * D, 1);
        var pos = Enumerable.Range(100, T).ToArray();
        var ff = Sharplight.Models.RopeScaling.Llama3(D, @base, 32f, 1f, 4f, 8192).Factors;
        Assert.Contains(ff, f => f > 1.5f);
        var tg = new TG(); var xin = tg.InF("x", TG.T, H, D); var pin = tg.In("pos", DataType.I32, TG.T, 1); var fin = tg.InF("ff", D / 2);
        tg.B.Output("y", tg.B.Emit1("Rope", [xin, pin, fin], Attributes.From(("dims", D), ("base", @base), ("style", "neox"))));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x); Kit.Set(m, pin, pos); Kit.Set(m, fin, ff);
        var p = Kit.P(T); m.Run(p);
        Kit.AssertClose(Ref.Rope(x, T, H, D, pos, 1, D, @base, "neox", null, false, 1f, ff), Kit.Out(m, "y", p), 1e-4, 1e-4);
    }

    /// <summary>Rope applied to a strided Slice view (as in Qwen3.5 q after the joint q/gate projection).</summary>
    [Fact]
    public void RopeOnStridedInput()
    {
        int T = 4, H = 2, D = 8;
        var full = Kit.Rand(T * H * 2 * D, 3); var pos = Enumerable.Range(10, T).ToArray();
        var tg = new TG(); var xin = tg.InF("x", TG.T, H * 2 * D); var pin = tg.In("pos", DataType.I32, TG.T, 1);
        var q = tg.B.Slice(tg.B.Reshape(xin, 0, H, 2 * D), 2, 0, D);
        tg.B.Output("y", tg.B.Emit1("Rope", [q, pin], Attributes.From(("dims", D), ("base", 1000f), ("style", "neox"))));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, full); Kit.Set(m, pin, pos);
        var p = Kit.P(T); m.Run(p);
        var qs = new float[T * H * D]; for (int r = 0; r < T * H; r++)
        {
            for (int i = 0; i < D; i++)
            {
                qs[r * D + i] = full[r * 2 * D + i];
            }
        }

        Kit.AssertClose(Ref.Rope(qs, T, H, D, pos, 1, D, 1000f, "neox", null, false, 1f), Kit.Out(m, "y", p), 1e-5, 1e-4);
    }
}
