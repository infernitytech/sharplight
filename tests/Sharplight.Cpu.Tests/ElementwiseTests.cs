using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

public class ElementwiseTests
{
    private static readonly Dictionary<string, Func<double, double>> UnaryRef = new()
    {
        ["Neg"] = x => -x, ["Exp"] = Math.Exp, ["Log"] = Math.Log, ["Sqrt"] = Math.Sqrt, ["Rsqrt"] = x => 1 / Math.Sqrt(x),
        ["Sigmoid"] = x => 1 / (1 + Math.Exp(-x)), ["Silu"] = x => x / (1 + Math.Exp(-x)),
        ["Gelu"] = x => 0.5 * x * (1 + ScalarMath.Erf(x / Math.Sqrt(2))),
        ["GeluTanh"] = x => 0.5 * x * (1 + Math.Tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))),
        ["Tanh"] = Math.Tanh, ["Softplus"] = x => x > 20 ? x : Math.Log(1 + Math.Exp(x)), ["Relu"] = x => Math.Max(0, x),
        ["Abs"] = Math.Abs, ["Square"] = x => x * x, ["Recip"] = x => 1 / x,
        ["Erf"] = ScalarMath.Erf, ["HardSwish"] = x => x * Math.Clamp(x / 6 + 0.5, 0, 1), ["HardSigmoid"] = x => Math.Clamp(0.2 * x + 0.5, 0, 1),
    };

    public static IEnumerable<object[]> UnaryNames() => StandardOps.UnaryNames.Where(n => UnaryRef.ContainsKey(n)).Select(n => new object[] { n });

    [Theory, MemberData(nameof(UnaryNames))]
    public void Unary(string op)
    {
        bool positive = op is "Log" or "Sqrt" or "Rsqrt" or "Recip";
        int T = 3;
        int n = 37;
        var x = Kit.Rand(T * n, 1, op == "Softplus" ? 30f : 6f);
        if (positive)
        {
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = Math.Abs(x[i]) + 0.1f;
            }
        }

        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, n);
        tg.B.Output("y", tg.B.Unary(op, xin));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(T);
        m.Run(p);
        Kit.AssertClose(x.Select(v => (float)UnaryRef[op](v)).ToArray(), Kit.Out(m, "y", p), 1e-5, 2e-5, op);
    }

    [Fact]
    public void LargeUnaryUsesParallelPath()
    {
        int T = 64;
        int n = 2048;
        var x = Kit.Rand(T * n, 2, 6f);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, n);
        tg.B.Output("y", tg.B.Silu(xin));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(T);
        m.Run(p);
        Kit.AssertClose(x.Select(v => Ref.Silu(v)).ToArray(), Kit.Out(m, "y", p), 1e-5, 2e-5);
    }

    [Fact]
    public void ErfAccuracy()
    {
        (double x, double e)[] t = [(0.1, 0.1124629160182849), (0.5, 0.5204998778130465), (1, 0.8427007929497149), (-1.5, -0.9661051464753108),
            (2, 0.9953222650189527), (2.5, 0.9995930479825550), (3, 0.9999779095030014), (4, 0.9999999845827421), (0, 0)];
        foreach (var (x, e) in t)
        {
            Assert.True(Math.Abs(ScalarMath.Erf(x) - e) < 1e-12, $"erf({x}) = {ScalarMath.Erf(x)} vs {e}");
        }
    }

    [Fact]
    public void ScaleOp()
    {
        var x = Kit.Rand(3 * 21, 3);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, 21);
        tg.B.Output("y", tg.B.Scale(xin, 2.5f, -0.25f));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(3);
        m.Run(p);
        Kit.AssertClose(x.Select(v => v * 2.5f - 0.25f).ToArray(), Kit.Out(m, "y", p));
    }

    private static float[] BcastRef(string op, float[] a, long[] ad, float[] b, long[] bd, out long[] od)
    {
        int r = Math.Max(ad.Length, bd.Length);
        od = new long[r];
        long[] pa = Enumerable.Repeat(1L, r - ad.Length).Concat(ad).ToArray();
        long[] pb = Enumerable.Repeat(1L, r - bd.Length).Concat(bd).ToArray();
        for (int i = 0; i < r; i++)
        {
            od[i] = Math.Max(pa[i], pb[i]);
        }

        long total = od.Aggregate(1L, (p, d) => p * d);
        var y = new float[total];
        for (long lin = 0; lin < total; lin++)
        {
            long rem = lin;
            long ia = 0;
            long ib = 0;
            long sa = 1;
            long sb = 1;
            for (int d = r - 1; d >= 0; d--)
            {
                long idx = rem % od[d];
                rem /= od[d];
                ia += (pa[d] == 1 ? 0 : idx) * sa;
                ib += (pb[d] == 1 ? 0 : idx) * sb;
                sa *= pa[d];
                sb *= pb[d];
            }
            float u = a[ia];
            float v = b[ib];
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
        long an = ad.Aggregate(1L, (p, d) => p * d);
        long bn = bd.Aggregate(1L, (p, d) => p * d);
        var a = Kit.Rand((int)an, 5);
        var b = Kit.Rand((int)bn, 6);
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

        var tg = new TestGraph();
        var ain = tg.InF("a", [TestGraph.T, .. ad.Skip(1).Select(d => (Dim)d)]);
        Value bv;
        Value? bin = null;
        if (bd[0] == T)
        {
            bin = tg.InF("b", [TestGraph.T, .. bd.Skip(1).Select(d => (Dim)d)]);
            bv = bin;
        }
        else
        {
            bv = tg.WF("b", b, bd);
        }

        tg.B.Output("y", tg.B.Binary(op, ain, bv));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, ain, a);
        if (bin != null)
        {
            Kit.Set(m, bin, b);
        }

        var p = Kit.P(T);
        m.Run(p);
        Kit.AssertClose(BcastRef(op, a, ad, b, bd, out _), Kit.Out(m, "y", p), 1e-6, 1e-6, $"{op} {name}");
    }

    [Fact]
    public void BinaryOnStridedViewAndWeightInF16()
    {
        // y = slice(x)[:, 8:40] * w16[32]
        int T = 4;
        int wide = 64;
        int n = 32;
        var x = Kit.Rand(T * wide, 7);
        var wf = Kit.Rand(n, 8);
        var w16 = new byte[n * 2];
        var wr = new float[n];
        for (int i = 0; i < n; i++)
        {
            var h = (Half)wf[i];
            BitConverter.TryWriteBytes(w16.AsSpan(i * 2), h);
            wr[i] = (float)h;
        }
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, wide);
        var w = tg.W("w", DataType.F16, new Shape(n), w16);
        tg.B.Output("y", tg.B.Mul(tg.B.Slice(xin, 1, 8, n), w));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(T);
        m.Run(p);
        var exp = new float[T * n];
        for (int t = 0; t < T; t++)
        {
            for (int i = 0; i < n; i++)
            {
                exp[t * n + i] = x[t * wide + 8 + i] * wr[i];
            }
        }

        Kit.AssertClose(exp, Kit.Out(m, "y", p));
    }
}
