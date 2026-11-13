using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

internal static class GK
{
    public static long[] Strides(long[] d) { var s = new long[d.Length]; long e = 1; for (int i = d.Length - 1; i >= 0; i--) { s[i] = e; e *= d[i]; } return s; }
    public static long[] Unravel(long l, long[] d) { var ix = new long[d.Length]; for (int i = d.Length - 1; i >= 0; i--) { ix[i] = l % d[i]; l /= d[i]; } return ix; }
    public static long P(long[] d) => d.Aggregate(1L, (a, b) => a * b);

    public static Value SetUp(TG tg, string name, DataType t, params long[] dims) => tg.In(name, t, dims.Select(d => (Dim)d).ToArray());

    public static void SetI64(CompiledModel m, Value v, long[] d) => m.SetInput(v, MemoryMarshal.AsBytes(d.AsSpan()));
    public static long[] OutL(CompiledModel m, string name, RunParameters p)
    {
        var v = m.Graph.Outputs[name];
        var r = new long[v.Shape.Elements(p.Symbols)];
        m.GetOutput<long>(name, r);
        return r;
    }

    public static Attributes A(params (string, object)[] kv) => Attributes.From(kv);
}

public class UnaryGeneralTests
{
    private static readonly Dictionary<string, Func<double, double>> Ref = new()
    {
        ["Sin"] = Math.Sin, ["Cos"] = Math.Cos, ["Tan"] = Math.Tan, ["Asin"] = Math.Asin, ["Acos"] = Math.Acos, ["Atan"] = Math.Atan,
        ["Sinh"] = Math.Sinh, ["Cosh"] = Math.Cosh, ["Asinh"] = Math.Asinh, ["Acosh"] = Math.Acosh, ["Atanh"] = Math.Atanh,
        ["Floor"] = Math.Floor, ["Ceil"] = Math.Ceiling, ["Round"] = x => Math.Round(x, MidpointRounding.ToEven), ["Sign"] = x => Math.Sign(x),
        ["Softsign"] = x => x / (1 + Math.Abs(x)), ["Mish"] = x => x * Math.Tanh(Math.Log(1 + Math.Exp(x))),
        ["Elu"] = x => x >= 0 ? x : Math.Exp(x) - 1, ["Celu"] = x => Math.Max(0, x) + Math.Min(0, Math.Exp(x) - 1),
        ["Selu"] = x => x > 0 ? 1.0507009873554805 * x : 1.0507009873554805 * 1.6732632423543772 * (Math.Exp(x) - 1),
        ["LeakyRelu"] = x => x >= 0 ? x : 0.01 * x, ["ThresholdedRelu"] = x => x > 1 ? x : 0, ["Not"] = x => x != 0 ? 0 : 1,
    };

    public static IEnumerable<object[]> Names() => Ref.Keys.Select(k => new object[] { k });

    private static float[] Run(string op, float[] x, Attributes? attrs = null)
    {
        var tg = new TG(); var xin = tg.InF("x", TG.T, x.Length / 3);
        tg.B.Output("y", tg.B.Emit1(op, [xin], attrs));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(3); m.Run(p);
        return Kit.Out(m, "y", p);
    }

    [Theory, MemberData(nameof(Names))]
    public void Unary(string op)
    {
        var x = Kit.Rand(3 * 41, 91, op is "Asin" or "Acos" or "Atanh" ? 0.95f : 4f);
        if (op == "Acosh")
        {
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = 1.01f + Math.Abs(x[i]) * 2;
            }
        }

        if (op == "Round") { x[0] = 0.5f; x[1] = 1.5f; x[2] = 2.5f; x[3] = -0.5f; x[4] = -1.5f; x[5] = -2.5f; x[6] = 2.4f; x[7] = 2.6f; }
        if (op == "Not") { x[0] = 0; x[1] = 2; x[2] = -3; }
        if (op == "Sign")
        {
            x[0] = 0;
        }

        Kit.AssertClose(x.Select(v => (float)Ref[op](v)).ToArray(), Run(op, x), 1e-5, 1e-5, op);
    }

    [Fact]
    public void AttributedActivations()
    {
        var x = Kit.Rand(3 * 41, 92, 3f);
        Kit.AssertClose(x.Select(v => v >= 0 ? v : 0.2f * v).ToArray(), Run("LeakyRelu", x, GK.A(("alpha", 0.2f))), 1e-6, 1e-6);
        Kit.AssertClose(x.Select(v => v > 0.5f ? v : 0f).ToArray(), Run("ThresholdedRelu", x, GK.A(("alpha", 0.5f))), 0, 0);
        Kit.AssertClose(x.Select(v => v >= 0 ? v : (float)(0.7 * (Math.Exp(v) - 1))).ToArray(), Run("Elu", x, GK.A(("alpha", 0.7f))), 1e-6, 1e-5);
        Kit.AssertClose(x.Select(v => (float)(Math.Max(0, v) + Math.Min(0, 0.7 * (Math.Exp(v / 0.7) - 1)))).ToArray(), Run("Celu", x, GK.A(("alpha", 0.7f))), 1e-6, 1e-5);
        Kit.AssertClose(x.Select(v => (float)(v > 0 ? 1.5 * v : 1.5 * 1.2 * (Math.Exp(v) - 1))).ToArray(), Run("Selu", x, GK.A(("alpha", 1.2f), ("gamma", 1.5f))), 1e-6, 1e-5);
    }

    [Fact]
    public void NanInfMasks()
    {
        var x = new float[] { float.NaN, 1f, float.PositiveInfinity, float.NegativeInfinity, 0f, -2f };
        var nan = new float[] { 1, 0, 0, 0, 0, 0 }; var inf = new float[] { 0, 0, 1, 1, 0, 0 };
        var tg = new TG(); var xin = tg.InF("x", TG.T, 3);
        tg.B.Output("nan", tg.B.Emit1("IsNaN", [xin])); tg.B.Output("inf", tg.B.Emit1("IsInf", [xin]));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(2); m.Run(p);
        Kit.AssertClose(nan, Kit.Out(m, "nan", p), 0, 0); Kit.AssertClose(inf, Kit.Out(m, "inf", p), 0, 0);
    }
}

public class BinaryGeneralTests
{
    private static float Fmod(float a, float b, bool fmod) { float r = (float)Math.IEEERemainder(0, 1); r = a % b; if (!fmod && r != 0 && (r < 0) != (b < 0)) { r += b; } return r; }

    [Theory]
    [InlineData("Mod", true), InlineData("Mod", false), InlineData("PRelu", false), InlineData("Equal", false), InlineData("NotEqual", false), InlineData("Less", false),
     InlineData("LessEqual", false), InlineData("Greater", false), InlineData("GreaterEqual", false), InlineData("And", false), InlineData("Or", false), InlineData("Xor", false)]
    public void Binary(string op, bool fmod)
    {
        int T = 3, n = 20;
        var a = Kit.Rand(T * n, 93, 5f); var b = Kit.Rand(n, 94, 3f);
        // make some values equal / zero / integral so comparisons and masks have ties
        for (int i = 0; i < b.Length; i += 3)
        {
            b[i] = a[i];
        }

        for (int i = 0; i < a.Length; i += 4)
        {
            a[i] = 0;
        }

        for (int i = 1; i < b.Length; i += 5)
        {
            b[i] = 0;
        }

        if (op == "Mod")
        {
            for (int i = 0; i < b.Length; i++)
            {
                if (b[i] == 0)
                {
                    b[i] = 1.7f;
                }
            }
        }

        var tg = new TG(); var ain = tg.InF("a", TG.T, n); var bv = tg.WF("b", b, n);
        tg.B.Output("y", tg.B.Emit1(op, [ain, bv], op == "Mod" ? GK.A(("fmod", fmod)) : null));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, ain, a);
        var p = Kit.P(T); m.Run(p);
        var exp = new float[T * n];
        for (int i = 0; i < exp.Length; i++)
        {
            float u = a[i], v = b[i % n];
            exp[i] = op switch
            {
                "Mod" => Fmod(u, v, fmod), "PRelu" => u < 0 ? u * v : u, "Equal" => u == v ? 1 : 0, "NotEqual" => u != v ? 1 : 0, "Less" => u < v ? 1 : 0,
                "LessEqual" => u <= v ? 1 : 0, "Greater" => u > v ? 1 : 0, "GreaterEqual" => u >= v ? 1 : 0, "And" => u != 0 && v != 0 ? 1 : 0,
                "Or" => u != 0 || v != 0 ? 1 : 0, _ => (u != 0) != (v != 0) ? 1 : 0,
            };
        }
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-6, 1e-6, op);
    }

    [Fact]
    public void ComparisonBroadcastAndPReluSlopePerChannel()
    {
        int N = 2, C = 3, H = 4, W = 5;
        var x = Kit.Rand(N * C * H * W, 95); var slope = Kit.Rand(C, 96);
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(C, H, W));
        tg.B.Output("p", tg.B.Emit1("PRelu", [xin, tg.WF("s", slope, C, 1, 1)]));
        tg.B.Output("g", tg.B.Emit1("Greater", [xin, tg.WF("z", [0f], 1, 1, 1)]));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2); Kit.Set(m, xin, x);
        var p = Kit.P(N); m.Run(p);
        Kit.AssertClose(x.Select((v, i) => v < 0 ? v * slope[i / (H * W) % C] : v).ToArray(), Kit.Out(m, "p", p), 1e-6, 1e-6);
        Kit.AssertClose(x.Select(v => v > 0 ? 1f : 0f).ToArray(), Kit.Out(m, "g", p), 0, 0);
    }
}

public class DataMovementTests
{
    [Fact]
    public void WhereBroadcastF32()
    {
        int N = 3, C = 4, W = 6;
        var x = Kit.Rand(N * C * W, 101); var y = Kit.Rand(C * W, 102); var cond = Kit.Rand(C * 1, 103).Select(v => v > 0 ? 1f : 0f).ToArray();
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(C, W));
        var mask = tg.B.Emit1("Greater", [xin, tg.WF("z", [0.1f], 1, 1)]);
        tg.B.Output("w1", tg.B.Emit1("Where", [mask, xin, tg.WF("y", y, C, W)]));
        tg.B.Output("w2", tg.B.Emit1("Where", [tg.WF("c", cond, C, 1), xin, tg.WF("s", [-7f], 1)]));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3); Kit.Set(m, xin, x);
        var p = Kit.P(N); m.Run(p);
        Kit.AssertClose(x.Select((v, i) => v > 0.1f ? v : y[i % (C * W)]).ToArray(), Kit.Out(m, "w1", p), 0, 0);
        Kit.AssertClose(x.Select((v, i) => cond[i / W % C] != 0 ? v : -7f).ToArray(), Kit.Out(m, "w2", p), 0, 0);
    }

    [Fact]
    public void WhereOnIntegerTensors()
    {
        int T = 3, n = 5;
        var a = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }; var cond = new float[] { 1, 0, 1, 0, 1 };
        var b64 = Enumerable.Range(0, T * n).Select(i => -(long)i * 5_000_000_000L).ToArray();
                var tg2 = new TG(); var a32 = tg2.In("a", DataType.I32, TG.T, n); var o32 = tg2.In("o", DataType.I32, TG.T, n); var c2 = tg2.InF("c", n);
        var a64 = tg2.In("a64", DataType.I64, TG.T, n); var o64 = tg2.In("o64", DataType.I64, TG.T, n);
        tg2.B.Output("w32", tg2.B.Emit1("Where", [c2, a32, o32])); tg2.B.Output("w64", tg2.B.Emit1("Where", [c2, a64, o64]));
        using var m = Kit.Compile(tg2.B.Build());
        var o = a.Select(v => -v).ToArray(); var l64 = a.Select(v => (long)v * 7_000_000_000L).ToArray();
        Kit.Set(m, a32, a); Kit.Set(m, o32, o); Kit.Set(m, c2, cond); GK.SetI64(m, a64, l64); GK.SetI64(m, o64, b64);
        var p = Kit.P(T); m.Run(p);
        Assert.Equal(a.Select((v, i) => cond[i % n] != 0 ? v : o[i]).ToArray(), Kit.OutI(m, "w32", p));
        Assert.Equal(l64.Select((v, i) => cond[i % n] != 0 ? v : b64[i]).ToArray(), GK.OutL(m, "w64", p));
    }

    [Fact]
    public void ExpandAndTile()
    {
        int N = 3, C = 4, W = 5;
        var x = Kit.Rand(N * C * 1, 104); var small = Kit.Rand(C * W, 105);
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(C, 1));
        tg.B.Output("e", tg.B.Emit1("Expand", [xin], GK.A(("shape", new long[] { 1, C, W }))));
        var sv = tg.WF("s", small, 1, C, W);
        tg.B.Output("e2", tg.B.Emit1("Expand", [sv], GK.A(("shape", new long[] { 2, 3, C, W }))));
        var xt = tg.InF("xt", SK.DSym(2, 3));
        tg.B.Output("t", tg.B.Emit1("Tile", [xt], GK.A(("repeats", new long[] { 1, 3, 2 }))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3); Kit.Set(m, xin, x);
        var xtv = Kit.Rand(N * 6, 106); Kit.Set(m, xt, xtv);
        var p = Kit.P(N); m.Run(p);
        var e = new float[N * C * W]; for (int i = 0; i < e.Length; i++)
        {
            e[i] = x[i / W];
        }

        Kit.AssertClose(e, Kit.Out(m, "e", p), 0, 0);
        var e2 = new float[2 * 3 * C * W]; for (int i = 0; i < e2.Length; i++)
        {
            e2[i] = small[i % (C * W)];
        }

        Kit.AssertClose(e2, Kit.Out(m, "e2", p), 0, 0);
        var t = new float[N * 6 * 6];
        for (int n = 0; n < N; n++)
        {
            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    t[(n * 6 + i) * 6 + j] = xtv[n * 6 + (i % 2) * 3 + j % 3];
                }
            }
        }

        Kit.AssertClose(t, Kit.Out(m, "t", p), 0, 0);
    }

    [Fact]
    public void TileAndExpandIntegers()
    {
        var tg = new TG(); var xi = tg.In("x", DataType.I64, 2, 3); var yi = tg.In("y", DataType.I32, 1, 3);
        tg.B.Output("t", tg.B.Emit1("Tile", [xi], GK.A(("repeats", new long[] { 2, 2 }))));
        tg.B.Output("e", tg.B.Emit1("Expand", [yi], GK.A(("shape", new long[] { 4, 3 }))));
        using var m = Kit.Compile(tg.B.Build());
        long[] x = [1, 2, 3, 40_000_000_000, 5, 6]; int[] y = [7, 8, 9];
        GK.SetI64(m, xi, x); Kit.Set(m, yi, y);
        var p = Kit.P(1); m.Run(p);
        var t = new long[4 * 6]; for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                t[i * 6 + j] = x[(i % 2) * 3 + j % 3];
            }
        }

        Assert.Equal(t, GK.OutL(m, "t", p));
        Assert.Equal(Enumerable.Range(0, 12).Select(i => y[i % 3]).ToArray(), Kit.OutI(m, "e", p));
    }

    public static IEnumerable<object[]> GatherAxisCases()
    {
        foreach (var t in new[] { DataType.I32, DataType.I64 })
        {
            foreach (var (axis, dims, idxDims) in new (int, long[], long[])[] { (0, [5, 4], [3]), (1, [3, 6], [2, 2]), (-1, [2, 3, 4], [5]), (1, [2, 5, 3], [4]), (0, [6], [2, 3]) })
            {
                yield return [t == DataType.I64, axis, dims, idxDims];
            }
        }
    }

    [Theory, MemberData(nameof(GatherAxisCases))]
    public void GatherAxis(bool idx64, int axis, long[] dims, long[] idxDims)
    {
        var data = Kit.Rand((int)GK.P(dims), 107); int ax = axis < 0 ? axis + dims.Length : axis;
        long A = dims[ax]; var rng = new Random(3);
        var idx = Enumerable.Range(0, (int)GK.P(idxDims)).Select(_ => (long)rng.Next((int)(2 * A)) - A).ToArray(); // includes negatives
        var tg = new TG(); var din = tg.InF("d", dims.Select(d => (Dim)d).ToArray());
        var iin = tg.In("i", idx64 ? DataType.I64 : DataType.I32, idxDims.Select(d => (Dim)d).ToArray());
        tg.B.Output("y", tg.B.Emit1("GatherAxis", [din, iin], GK.A(("axis", (long)axis))));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, din, data);
        if (idx64)
        {
            GK.SetI64(m, iin, idx);
        }
        else
        {
            Kit.Set(m, iin, idx.Select(v => (int)v).ToArray());
        }

        var p = Kit.P(1); m.Run(p);
        long outer = GK.P(dims[..ax]), inner = GK.P(dims[(ax + 1)..]);
        var exp = new List<float>();
        for (long o = 0; o < outer; o++)
        {
            foreach (var id in idx) { long ii = id < 0 ? id + A : id; for (long i = 0; i < inner; i++)
                {
                    exp.Add(data[(o * A + ii) * inner + i]);
                }
            }
        }

        Kit.AssertClose(exp.ToArray(), Kit.Out(m, "y", p), 0, 0);
    }

    [Fact]
    public void GatherAxisIntegerDataAndSymbolicIndices()
    {
        var tg = new TG(); var d = tg.In("d", DataType.I64, 6); var i = tg.In("i", DataType.I32, TG.T);
        tg.B.Output("y", tg.B.Emit1("GatherAxis", [d, i], GK.A(("axis", 0L))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 8);
        long[] data = [10, 20, 30, 40_000_000_000, 50, 60]; GK.SetI64(m, d, data); Kit.Set(m, i, new[] { 3, -1, 0 });
        var p = Kit.P(3); m.Run(p);
        Assert.Equal(new long[] { 40_000_000_000, 60, 10 }, GK.OutL(m, "y", p));
    }

    [Fact]
    public void GatherElementsMatchesReference()
    {
        long[] dd = [3, 4, 5], id = [3, 2, 5]; int axis = 1;
        var data = Kit.Rand((int)GK.P(dd), 108); var rng = new Random(5);
        var idx = Enumerable.Range(0, (int)GK.P(id)).Select(_ => rng.Next(-4, 4)).ToArray();
        var tg = new TG(); var din = tg.InF("d", dd.Select(d => (Dim)d).ToArray()); var iin = tg.In("i", DataType.I32, id.Select(d => (Dim)d).ToArray());
        tg.B.Output("y", tg.B.Emit1("GatherElements", [din, iin], GK.A(("axis", (long)axis))));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, din, data); Kit.Set(m, iin, idx);
        var p = Kit.P(1); m.Run(p);
        var ds = GK.Strides(dd); var exp = new float[idx.Length];
        for (int l = 0; l < exp.Length; l++) { var ix = GK.Unravel(l, id); long k = idx[l] < 0 ? idx[l] + dd[axis] : idx[l]; ix[axis] = k; exp[l] = data[ix.Zip(ds, (a, b) => a * b).Sum()]; }
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 0, 0);
    }

    [Theory, InlineData(0), InlineData(1)]
    public void GatherNd(int batchDims)
    {
        long[] dd = [3, 4, 5]; long[] id = batchDims == 0 ? [2, 3, 2] : [3, 2, 2];   // batch_dims=1: idx [3, 2, q=2] indexes data dims 1..2
        var data = Kit.Rand((int)GK.P(dd), 109); var rng = new Random(6);
        var idx = new int[(int)GK.P(id)];
        for (int l = 0; l < idx.Length; l++)
        {
            idx[l] = rng.Next(-3, 3) switch { var v when v < 0 => v, var v => v };
        }

        var tg = new TG(); var din = tg.InF("d", dd.Select(d => (Dim)d).ToArray()); var iin = tg.In("i", DataType.I32, id.Select(d => (Dim)d).ToArray());
        tg.B.Output("y", tg.B.Emit1("GatherND", [din, iin], GK.A(("batch_dims", (long)batchDims))));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, din, data); Kit.Set(m, iin, idx);
        var p = Kit.P(1); m.Run(p);
        var ds = GK.Strides(dd); int q = (int)id[^1]; long tuples = GK.P(id[..^1]); long slice = GK.P(dd[(batchDims + q)..]);
        long perBatch = GK.P(dd[batchDims..]); long tpb = GK.P(id[batchDims..^1]);
        var exp = new List<float>();
        for (long t = 0; t < tuples; t++)
        {
            long off = t / tpb * perBatch;
            for (int j = 0; j < q; j++) { long v = idx[t * q + j]; if (v < 0) { v += dd[batchDims + j]; } off += v * ds[batchDims + j]; }
            for (long i = 0; i < slice; i++)
            {
                exp.Add(data[off + i]);
            }
        }
        Kit.AssertClose(exp.ToArray(), Kit.Out(m, "y", p), 0, 0);
    }

    private static float Red(string r, float a, float b) => r switch { "add" => a + b, "mul" => a * b, "max" => Math.Max(a, b), "min" => Math.Min(a, b), _ => b };

    [Theory, InlineData("none"), InlineData("add"), InlineData("mul"), InlineData("max"), InlineData("min")]
    public void ScatterElements(string red)
    {
        long[] dd = [4, 5], id = [3, 5]; int axis = 0;
        var data = Kit.Rand(20, 110); var upd = Kit.Rand(15, 111);
        var idx = new int[15]; for (int i = 0; i < 15; i++)
        {
            idx[i] = (i * 7 + 1) % 4 - (i % 5 == 0 ? 4 : 0); // duplicates and negatives
        }

        var tg = new TG(); var din = tg.InF("d", 4, 5); var iin = tg.In("i", DataType.I32, 3, 5); var uin = tg.InF("u", 3, 5);
        tg.B.Output("y", tg.B.Emit1("ScatterElements", [din, iin, uin], GK.A(("axis", (long)axis), ("reduction", red))));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, din, data); Kit.Set(m, iin, idx); Kit.Set(m, uin, upd);
        var p = Kit.P(1); m.Run(p);
        var exp = (float[])data.Clone();
        for (int l = 0; l < 15; l++) { var ix = GK.Unravel(l, id); ix[axis] = idx[l] < 0 ? idx[l] + 4 : idx[l]; long o = ix[0] * 5 + ix[1]; exp[o] = Red(red, exp[o], upd[l]); }
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-6, 1e-6, red);
    }

    [Theory, InlineData("none"), InlineData("add"), InlineData("max")]
    public void ScatterNd(string red)
    {
        var data = Kit.Rand(4 * 3 * 2, 112); var upd = Kit.Rand(3 * 2 * 2, 113);
        int[] idx = [0, 1, 3, 2, 0, 1];  // [3, 2]: rows (0,1), (3,2), (0,1) duplicate
        var tg = new TG(); var din = tg.InF("d", 4, 3, 2); var iin = tg.In("i", DataType.I32, 3, 2); var uin = tg.InF("u", 3, 2);
        tg.B.Output("y", tg.B.Emit1("ScatterND", [din, iin, uin], GK.A(("reduction", red))));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, din, data); Kit.Set(m, iin, idx); Kit.Set(m, uin, upd.Take(6).ToArray());
        var p = Kit.P(1); m.Run(p);
        var exp = (float[])data.Clone();
        for (int t = 0; t < 3; t++)
        {
            for (int i = 0; i < 2; i++) { long o = (idx[t * 2] * 3 + idx[t * 2 + 1]) * 2 + i; exp[o] = Red(red, exp[o], upd[t * 2 + i]); }
        }

        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-6, 1e-6, red);
    }

    [Theory, InlineData(false, false), InlineData(true, false), InlineData(false, true), InlineData(true, true)]
    public void CumSum(bool exclusive, bool reverse)
    {
        long[] dims = [3, 5, 4];
        foreach (int axis in new[] { 0, 1, 2 })
        {
            var x = Kit.Rand((int)GK.P(dims), 114);
            var tg = new TG(); var xin = tg.InF("x", dims.Select(d => (Dim)d).ToArray());
            tg.B.Output("y", tg.B.Emit1("CumSum", [xin], GK.A(("axis", (long)axis), ("exclusive", exclusive), ("reverse", reverse))));
            using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
            var p = Kit.P(1); m.Run(p);
            var ds = GK.Strides(dims); var exp = new float[x.Length];
            for (int l = 0; l < x.Length; l++)
            {
                var ix = GK.Unravel(l, dims); double s = 0; long n = dims[axis];
                for (long k = 0; k < n; k++)
                {
                    bool inc = reverse ? k >= ix[axis] : k <= ix[axis];
                    if (exclusive && k == ix[axis])
                    {
                        inc = false;
                    }

                    if (!inc)
                    {
                        continue;
                    }

                    ix[axis] = GK.Unravel(l, dims)[axis]; var jx = (long[])ix.Clone(); jx[axis] = k;
                    s += x[jx.Zip(ds, (a, b) => a * b).Sum()];
                }
                exp[l] = (float)s;
            }
            Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-5, 1e-5, $"axis {axis}");
        }
    }

    [Theory, InlineData("max", 1, true, false), InlineData("min", 0, false, false), InlineData("max", 2, true, true), InlineData("min", 1, true, true), InlineData("max", -1, false, true)]
    public void ArgReduce(string mode, int axis, bool keep, bool last)
    {
        long[] dims = [4, 6, 5];
        var x = Kit.Rand((int)GK.P(dims), 115).Select(v => (float)Math.Round(v * 3)).ToArray();  // many ties
        var tg = new TG(); var xin = tg.InF("x", dims.Select(d => (Dim)d).ToArray());
        tg.B.Output("y", tg.B.Emit1("ArgReduce", [xin], GK.A(("mode", mode), ("axis", (long)axis), ("keepdims", keep), ("select_last_index", last))));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(1); m.Run(p);
        int ax = axis < 0 ? axis + 3 : axis; var ds = GK.Strides(dims);
        var od = dims.Where((_, i) => i != ax).ToArray(); var exp = new int[GK.P(od)];
        for (int l = 0; l < exp.Length; l++)
        {
            var ix2 = GK.Unravel(l, od); var ix = new long[3]; for (int d = 0, k = 0; d < 3; d++)
            {
                ix[d] = d == ax ? 0 : ix2[k++];
            }

            int best = 0; float bv = 0;
            for (int a = 0; a < dims[ax]; a++)
            {
                ix[ax] = a; float v = x[ix.Zip(ds, (u, w) => u * w).Sum()];
                if (a == 0) { bv = v; continue; }
                bool better = mode == "max" ? (last ? v >= bv : v > bv) : (last ? v <= bv : v < bv);
                if (better) { bv = v; best = a; }
            }
            exp[l] = best;
        }
        Assert.Equal(exp, Kit.OutI(m, "y", p));
        Assert.Equal(keep ? 3 : 2, m.Graph.Outputs["y"].Shape.Rank);
    }

    [Theory, InlineData(true, 3, -1), InlineData(false, 3, -1), InlineData(true, 6, -1), InlineData(true, 2, 0), InlineData(false, 4, 1)]
    public void TopK(bool largest, int k, int axis)
    {
        long[] dims = [5, 6, 7];
        var x = Kit.Rand((int)GK.P(dims), 116).Select(v => (float)Math.Round(v * 4)).ToArray();  // ties
        int ax = axis < 0 ? axis + 3 : axis; if (k > dims[ax])
        {
            k = (int)dims[ax];
        }

        var tg = new TG(); var xin = tg.InF("x", dims.Select(d => (Dim)d).ToArray());
        var outs = tg.B.Emit("TopK", [xin], GK.A(("k", (long)k), ("axis", (long)axis), ("largest", largest)));
        tg.B.Output("v", outs[0]); tg.B.Output("i", outs[1]);
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(1); m.Run(p);
        var ds = GK.Strides(dims); var od = (long[])dims.Clone(); od[ax] = k;
        var ev = new float[GK.P(od)]; var ei = new int[ev.Length];
        var os = GK.Strides(od);
        for (long l = 0; l < GK.P(dims) / dims[ax]; l++)
        {
            var rest = dims.Where((_, i) => i != ax).ToArray(); var r2 = GK.Unravel(l, rest); var ix = new long[3]; for (int d = 0, q = 0; d < 3; d++)
            {
                ix[d] = d == ax ? 0 : r2[q++];
            }

            var order = Enumerable.Range(0, (int)dims[ax]).Select(a => { ix[ax] = a; return (a, v: x[ix.Zip(ds, (u, w) => u * w).Sum()]); })
                .OrderBy(t => largest ? -t.v : t.v).ThenBy(t => t.a).Take(k).ToArray();
            for (int j = 0; j < k; j++) { ix[ax] = j; long o = ix.Zip(os, (u, w) => u * w).Sum(); ev[o] = order[j].v; ei[o] = order[j].a; }
        }
        Kit.AssertClose(ev, Kit.Out(m, "v", p), 0, 0); Assert.Equal(ei, Kit.OutI(m, "i", p));
    }

    [Theory, InlineData(true, 0), InlineData(false, 0), InlineData(true, 1), InlineData(false, -1), InlineData(true, 3)]
    public void Trilu(bool upper, int k)
    {
        var x = Kit.Rand(2 * 5 * 6, 117);
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(5, 6));
        tg.B.Output("y", tg.B.Emit1("Trilu", [xin], GK.A(("upper", upper), ("k", (long)k))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2); Kit.Set(m, xin, x);
        var p = Kit.P(2); m.Run(p);
        var exp = x.Select((v, l) => { int i = l / 6 % 5, j = l % 6; return (upper ? j - i >= k : j - i <= k) ? v : 0f; }).ToArray();
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 0, 0);
    }

    [Theory, InlineData(-1), InlineData(0), InlineData(1)]
    public void OneHot(int axis)
    {
        int[] idx = [2, 0, -1, 3, 7, 1]; // -1 wraps to depth-1; 7 is out of range -> all off
        var tg = new TG(); var iin = tg.In("i", DataType.I32, 2, 3);
        tg.B.Output("y", tg.B.Emit1("OneHot", [iin], GK.A(("depth", 4L), ("axis", (long)axis), ("on", 5f), ("off", -1f))));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, iin, idx);
        var p = Kit.P(1); m.Run(p);
        int ax = axis < 0 ? axis + 3 : axis; long[] od = ax == 0 ? [4, 2, 3] : ax == 1 ? [2, 4, 3] : [2, 3, 4];
        var exp = new float[24];
        for (int l = 0; l < 24; l++)
        {
            var ix = GK.Unravel(l, od); int d = (int)ix[ax]; var rest = ix.Where((_, i) => i != ax).ToArray();
            int v = idx[rest[0] * 3 + rest[1]]; if (v < 0)
            {
                v += 4;
            }

            exp[l] = v == d ? 5f : -1f;
        }
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 0, 0);
    }
}

public class NormReduceGeneralTests
{
    [Theory]
    [InlineData(1, 6, true, true), InlineData(2, 6, true, true), InlineData(6, 6, true, true), InlineData(3, 6, false, false), InlineData(2, 8, true, false)]
    public void GroupNorm(int groups, int C, bool scale, bool bias)
    {
        int N = 3, H = 5, W = 7;
        var x = Kit.Rand(N * C * H * W, 121, 2f); var sc = Kit.Rand(C, 122); var bi = Kit.Rand(C, 123);
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(C, H, W));
        var ins = new List<Value> { xin }; if (scale)
        {
            ins.Add(tg.WF("s", sc, C));
        }

        if (bias) { if (!scale) { ins.Add(tg.WF("s1", Enumerable.Repeat(1f, C).ToArray(), C)); } ins.Add(tg.WF("b", bi, C)); }
        tg.B.Output("y", tg.B.Emit1("GroupNorm", ins, GK.A(("groups", (long)groups), ("eps", 1e-5f))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3); Kit.Set(m, xin, x);
        var p = Kit.P(N); m.Run(p);
        var exp = new float[x.Length]; int cg = C / groups, S = H * W;
        for (int n = 0; n < N; n++)
        {
            for (int g = 0; g < groups; g++)
        {
            double mean = 0, v = 0; long cnt = cg * S;
            for (int i = 0; i < cnt; i++)
                {
                    mean += x[(n * C + g * cg) * S + i];
                }

                mean /= cnt;
            for (int i = 0; i < cnt; i++)
                {
                    v += Math.Pow(x[(n * C + g * cg) * S + i] - mean, 2);
                }

                v /= cnt;
            for (int c = 0; c < cg; c++)
                {
                    for (int s = 0; s < S; s++)
            {
                int ch = g * cg + c; long o = (n * C + ch) * S + s;
                exp[o] = (float)((x[o] - mean) / Math.Sqrt(v + 1e-5) * (scale ? sc[ch] : 1) + (bias ? bi[ch] : 0));
            }
                }
            }
        }

        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4);
    }

    [Fact]
    public void GroupNormRank3()
    {
        int N = 2, C = 4, L = 9; var x = Kit.Rand(N * C * L, 124);
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(C, L));
        tg.B.Output("y", tg.B.Emit1("GroupNorm", [xin], GK.A(("groups", 2L))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2); Kit.Set(m, xin, x);
        var p = Kit.P(N); m.Run(p);
        var exp = new float[x.Length];
        for (int n = 0; n < N; n++)
        {
            for (int g = 0; g < 2; g++)
        {
            int o0 = (n * C + g * 2) * L, cnt = 2 * L; double mean = x.Skip(o0).Take(cnt).Average(v => (double)v), var = x.Skip(o0).Take(cnt).Average(v => Math.Pow(v - mean, 2));
            for (int i = 0; i < cnt; i++)
                {
                    exp[o0 + i] = (float)((x[o0 + i] - mean) / Math.Sqrt(var + 1e-5));
                }
            }
        }

        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4);
    }

    public static IEnumerable<object[]> ReduceModes() { foreach (var mode in new[] { "prod", "l1", "l2", "sumsquare", "logsumexp" })
        {
            foreach (var (axes, keep, perm) in new (long[], bool, bool)[] { ([1], true, false), ([2, 3], false, false), ([0, 2], true, false), ([1, 2, 3], false, false), ([2, 3], true, true), ([1], false, true) })
            {
                yield return [mode, axes, keep, perm];
            }
        }
    }

    [Theory, MemberData(nameof(ReduceModes))]
    public void ReduceNewModes(string mode, long[] axes, bool keep, bool permuted)
    {
        long[] dims = [3, 4, 5, 6]; long total = GK.P(dims);
        var x = Kit.Rand((int)total, 125, mode == "prod" ? 1.3f : 2f);
        var tg = new TG(); Value src, xin; float[] logical = x;
        if (permuted)
        {
            xin = tg.InF("x", SK.D(3, 5, 6, 4)); src = tg.B.Permute(xin, 0, 3, 1, 2);
            logical = new float[total];
            for (int n = 0; n < 3; n++)
            {
                for (int h = 0; h < 5; h++)
                {
                    for (int w = 0; w < 6; w++)
                    {
                        for (int c = 0; c < 4; c++)
                        {
                            logical[((n * 4 + c) * 5 + h) * 6 + w] = x[((n * 5 + h) * 6 + w) * 4 + c];
                        }
                    }
                }
            }
        }
        else { xin = tg.InF("x", SK.D(dims)); src = xin; }
        tg.B.Output("y", tg.B.Emit1("Reduce", [src], GK.A(("mode", mode), ("axes", axes), ("keepdims", keep))));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(1); m.Run(p);
        // reference through per-mode accumulation using Sum/Max reference of transformed data
        var red = new bool[4]; foreach (var a in axes)
        {
            red[a] = true;
        }

        var keptDims = dims.Where((_, i) => !red[i]).ToArray(); var exp = new float[GK.P(keptDims)];
        for (int o = 0; o < exp.Length; o++)
        {
            var kix = GK.Unravel(o, keptDims); var vals = new List<double>();
            for (long l = 0; l < total; l++)
            {
                var ix = GK.Unravel(l, dims); bool ok = true; for (int d = 0, q = 0; d < 4; d++)
                {
                    if (!red[d]) { if (ix[d] != kix[q++]) { ok = false; break; } }
                }

                if (ok)
                {
                    vals.Add(logical[l]);
                }
            }
            exp[o] = (float)(mode switch
            {
                "prod" => vals.Aggregate(1.0, (a, b) => a * b), "l1" => vals.Sum(Math.Abs), "l2" => Math.Sqrt(vals.Sum(v => v * v)), "sumsquare" => vals.Sum(v => v * v),
                _ => Math.Log(vals.Sum(Math.Exp)),
            });
        }
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4, mode);
        Assert.Equal(keep ? 4 : 4 - axes.Length, m.Graph.Outputs["y"].Shape.Rank);
    }

    [Fact]
    public void LogSoftmax()
    {
        int T = 3, n = 19; var x = Kit.Rand(T * n, 126, 4f);
        var tg = new TG(); var xin = tg.InF("x", TG.T, n);
        tg.B.Output("y", tg.B.Emit1("Softmax", [xin], GK.A(("log", true), ("scale", 0.8f))));
        tg.B.Output("y2", tg.B.Emit1("Softmax", [xin], GK.A(("log", true))));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(T); m.Run(p);
        var sm = Ref.Softmax(x, T, n, 0.8f); var sm2 = Ref.Softmax(x, T, n, 1f);
        Kit.AssertClose(sm.Select(v => (float)Math.Log(v)).ToArray(), Kit.Out(m, "y", p), 1e-5, 1e-5);
        Kit.AssertClose(sm2.Select(v => (float)Math.Log(v)).ToArray(), Kit.Out(m, "y2", p), 1e-5, 1e-5);
    }
}

public class ConvTransposeTests
{
    public record Case(string Name, int C, int H, int W, int O, int KH, int KW, int[] S, int[] P, int[] D, int[] OP, int G, bool Bias);
    private static readonly Case[] Cases =
    [
        new("k2s2", 4, 5, 6, 3, 2, 2, [2, 2], [0, 0, 0, 0], [1, 1], [0, 0], 1, true),
        new("k3s2p1op1", 5, 6, 7, 4, 3, 3, [2, 2], [1, 1, 1, 1], [1, 1], [1, 1], 1, true),
        new("k4s2p1", 6, 8, 8, 5, 4, 4, [2, 2], [1, 1, 1, 1], [1, 1], [0, 0], 1, false),
        new("k3s1p1", 3, 9, 10, 7, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], [0, 0], 1, true),
        new("k3s1", 3, 4, 5, 2, 3, 3, [1, 1], [0, 0, 0, 0], [1, 1], [0, 0], 1, false),
        new("dil2", 4, 6, 6, 3, 3, 3, [1, 1], [2, 2, 2, 2], [2, 2], [0, 0], 1, true),
        new("groups2", 6, 5, 5, 8, 3, 3, [2, 2], [1, 1, 1, 1], [1, 1], [1, 1], 2, true),
        new("asym", 4, 5, 7, 6, 3, 2, [2, 1], [1, 0, 0, 1], [1, 1], [0, 0], 2, true),
        new("longK", 300, 3, 4, 5, 3, 3, [2, 2], [1, 1, 1, 1], [1, 1], [1, 1], 1, true),
        new("rect-s3", 4, 4, 5, 3, 3, 3, [3, 3], [0, 0, 0, 0], [1, 1], [0, 0], 1, true),
    ];
    public static IEnumerable<object[]> Params() { foreach (var c in Cases)
        {
            foreach (var n in new[] { 1, 3 })
            {
                yield return [c.Name, n];
            }
        }
    }

    [Theory, MemberData(nameof(Params))]
    public void ConvTranspose(string name, int N)
    {
        var c = Cases.Single(k => k.Name == name); int Og = c.O / c.G, Cg = c.C / c.G;
        var x = Kit.Rand(N * c.C * c.H * c.W, 131); var w = Kit.Rand(c.C * Og * c.KH * c.KW, 132, 0.5f); var bias = c.Bias ? Kit.Rand(c.O, 133) : null;
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(c.C, c.H, c.W));
        var ins = new List<Value> { xin, tg.WF("w", w, c.C, Og, c.KH, c.KW) }; if (bias != null)
        {
            ins.Add(tg.WF("b", bias, c.O));
        }

        long[] L(int[] v) => v.Select(q => (long)q).ToArray();
        tg.B.Output("y", tg.B.Emit1("ConvTranspose", ins, GK.A(("groups", (long)c.G), ("strides", L(c.S)), ("pads", L(c.P)), ("dilations", L(c.D)), ("output_padding", L(c.OP)))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3); Kit.Set(m, xin, x);
        var p = Kit.P(N); m.Run(p);
        int HO = (c.H - 1) * c.S[0] - c.P[0] - c.P[2] + c.D[0] * (c.KH - 1) + c.OP[0] + 1, WO = (c.W - 1) * c.S[1] - c.P[1] - c.P[3] + c.D[1] * (c.KW - 1) + c.OP[1] + 1;
        var exp = new double[N * c.O * HO * WO];
        for (int n = 0; n < N; n++)
        {
            for (int o = 0; o < c.O; o++)
            {
                for (int i = 0; i < HO * WO; i++)
                {
                    exp[(n * c.O + o) * HO * WO + i] = bias?[o] ?? 0;
                }
            }
        }

        for (int n = 0; n < N; n++)
        {
            for (int g = 0; g < c.G; g++)
            {
                for (int cc = 0; cc < Cg; cc++)
                {
                    for (int o = 0; o < Og; o++)
                    {
                        for (int ih = 0; ih < c.H; ih++)
                        {
                            for (int iw = 0; iw < c.W; iw++)
            {
                double xv = x[((n * c.C + g * Cg + cc) * c.H + ih) * c.W + iw];
                for (int kh = 0; kh < c.KH; kh++)
                                {
                                    for (int kw = 0; kw < c.KW; kw++)
                {
                    int oh = ih * c.S[0] - c.P[0] + kh * c.D[0], ow = iw * c.S[1] - c.P[1] + kw * c.D[1];
                    if (oh < 0 || oh >= HO || ow < 0 || ow >= WO)
                                        {
                                            continue;
                                        }

                                        exp[((n * c.O + g * Og + o) * HO + oh) * WO + ow] += xv * w[(((g * Cg + cc) * Og + o) * c.KH + kh) * c.KW + kw];
                }
                                }
                            }
                        }
                    }
                }
            }
        }

        var shape = m.Graph.Outputs["y"].Shape; Assert.Equal(HO, (int)shape[2].Fixed); Assert.Equal(WO, (int)shape[3].Fixed);
        Kit.AssertClose(exp.Select(v => (float)v).ToArray(), Kit.Out(m, "y", p), 1e-4, 1e-4, name);
    }

    [Fact]
    public void ConvTransposeF16Weights()
    {
        int C = 4, O = 3, H = 5, W = 5; var x = Kit.Rand(C * H * W, 134); var wf = Kit.Rand(C * O * 4, 135, 0.5f);
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(C, H, W));
        var wv = SK.Weight(tg, "w", wf, [C, O, 2, 2], "f16", out var weff);
        tg.B.Output("y", tg.B.Emit1("ConvTranspose", [xin, wv], GK.A(("strides", new long[] { 2, 2 }))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 1); Kit.Set(m, xin, x);
        var p = Kit.P(1); m.Run(p);
        var exp = new float[O * 10 * 10];
        for (int c = 0; c < C; c++)
        {
            for (int o = 0; o < O; o++)
            {
                for (int ih = 0; ih < H; ih++)
                {
                    for (int iw = 0; iw < W; iw++)
                    {
                        for (int kh = 0; kh < 2; kh++)
                        {
                            for (int kw = 0; kw < 2; kw++)
                            {
                                exp[(o * 10 + ih * 2 + kh) * 10 + iw * 2 + kw] += x[(c * H + ih) * W + iw] * weff[((c * O + o) * 2 + kh) * 2 + kw];
                            }
                        }
                    }
                }
            }
        }

        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4);
    }
}

public class CastAndIntegerFlowTests
{
    [Fact]
    public void CastBetweenTypes()
    {
        float[] x = [0f, 1f, -1f, 2.7f, -2.7f, 1000f, 65504f, 0.5f];
        var tg = new TG(); var xin = tg.InF("x", TG.T, 4);
        foreach (var t in new[] { "f16", "i32", "i64", "bool" })
        {
            tg.B.Output(t, tg.B.Emit1("Cast", [xin], GK.A(("type", t))));
        }

        var f16 = tg.B.Emit1("Cast", [xin], GK.A(("type", "f16")));
        tg.B.Output("f16_back", tg.B.Emit1("Cast", [f16], GK.A(("type", "f32"))));
        tg.B.Output("i64_back", tg.B.Emit1("Cast", [tg.B.Emit1("Cast", [xin], GK.A(("type", "i64")))], GK.A(("type", "f32"))));
        tg.B.Output("i32_i64", tg.B.Emit1("Cast", [tg.B.Emit1("Cast", [xin], GK.A(("type", "i32")))], GK.A(("type", "i64"))));
        tg.B.Output("bool_back", tg.B.Emit1("Cast", [tg.B.Emit1("Cast", [xin], GK.A(("type", "bool")))], GK.A(("type", "f32"))));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(2); m.Run(p);
        Kit.AssertClose(x.Select(v => (float)(Half)v).ToArray(), Kit.Out(m, "f16_back", p), 0, 0);
        Kit.AssertClose(x.Select(v => (float)(long)v).ToArray(), Kit.Out(m, "i64_back", p), 0, 0);
        Assert.Equal(x.Select(v => (long)(int)v).ToArray(), GK.OutL(m, "i32_i64", p));
        Kit.AssertClose(x.Select(v => v != 0 ? 1f : 0f).ToArray(), Kit.Out(m, "bool_back", p), 0, 0);
        Assert.Equal(x.Select(v => (int)v).ToArray(), Kit.OutI(m, "i32", p));
    }

    [Fact]
    public void IntegerTensorsFlowThroughViewsAndData()
    {
        // i64 shape-like tensor: reshape -> slice -> concat -> gather -> expand
        var tg = new TG(); var x = tg.In("x", DataType.I64, 2, 6);
        var r = tg.B.Reshape(x, 3, 4);                       // [3,4]
        var s1 = tg.B.Slice(r, 1, 1, 2); var s2 = tg.B.Slice(r, 1, 0, 2);
        var cc = tg.B.Concat(1, s1, s2);                     // [3,4]
        var iv = tg.In("i", DataType.I32, 2);
        var g = tg.B.Emit1("GatherAxis", [cc, iv], GK.A(("axis", 0L)));   // [2,4]
        tg.B.Output("g", g);
        tg.B.Output("e", tg.B.Emit1("Expand", [tg.B.Slice(g, 0, 0, 1)], GK.A(("shape", new long[] { 3, 4 }))));
        tg.B.Output("perm", tg.B.Contiguous(tg.B.Permute(cc, 1, 0)));
        using var m = Kit.Compile(tg.B.Build());
        long[] data = Enumerable.Range(0, 12).Select(i => (long)i * 3_000_000_000L).ToArray();
        GK.SetI64(m, x, data); Kit.Set(m, iv, new[] { 2, 0 });
        var p = Kit.P(1); m.Run(p);
        var cat = new long[12]; for (int i = 0; i < 3; i++) { cat[i * 4] = data[i * 4 + 1]; cat[i * 4 + 1] = data[i * 4 + 2]; cat[i * 4 + 2] = data[i * 4]; cat[i * 4 + 3] = data[i * 4 + 1]; }
        var gexp = new long[8]; for (int j = 0; j < 4; j++) { gexp[j] = cat[2 * 4 + j]; gexp[4 + j] = cat[j]; }
        Assert.Equal(gexp, GK.OutL(m, "g", p));
        Assert.Equal(Enumerable.Range(0, 12).Select(i => gexp[i % 4]).ToArray(), GK.OutL(m, "e", p));
        var perm = new long[12]; for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                perm[j * 3 + i] = cat[i * 4 + j];
            }
        }

        Assert.Equal(perm, GK.OutL(m, "perm", p));
    }
}

/// <summary>Writes inputs/outputs of selected ops to $SHARPLIGHT_ORT_DUMP for the onnxruntime comparison script (no-op otherwise).</summary>
public class OrtDumpGeneralTests
{
    private static void Save(string dir, string name, string suffix, Array a) =>
        File.WriteAllBytes(Path.Combine(dir, $"{name}.{suffix}.bin"), MemoryMarshal.AsBytes(a is float[] f ? f.AsSpan() : a is int[] i ? i.AsSpan().Cast() : Span<float>.Empty).ToArray());

    [Fact]
    public void Dump()
    {
        var dir = Environment.GetEnvironmentVariable("SHARPLIGHT_ORT_DUMP");
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }

        dir = Path.Combine(dir, "general"); Directory.CreateDirectory(dir);
        void Run(string name, Func<TG, (Value[] ins, Value[] outs)> build, params Array[] inputs)
        {
            var tg = new TG(); var (ins, outs) = build(tg);
            for (int i = 0; i < outs.Length; i++)
            {
                tg.B.Output("o" + i, outs[i]);
            }

            using var m = Kit.Compile(tg.B.Build());
            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] is float[] f)
                {
                    Kit.Set(m, ins[i], f);
                }
                else
                {
                    Kit.Set(m, ins[i], (int[])inputs[i]);
                }

                Save(dir, name, "in" + i, inputs[i]);
            }
            var p = Kit.P(1); m.Run(p);
            for (int i = 0; i < outs.Length; i++)
            {
                if (outs[i].Type == DataType.I32)
                {
                    Save(dir, name, "out" + i, Kit.OutI(m, "o" + i, p));
                }
                else
                {
                    Save(dir, name, "out" + i, Kit.Out(m, "o" + i, p));
                }
            }
        }
        foreach (var mode in new[] { "linear", "nearest", "cubic" })
        {
            foreach (var pad in new[] { "zeros", "border", "reflection" })
            {
                foreach (var ac in new[] { false, true })
        {
            var x = Kit.Rand(2 * 5 * 6, 201); var g = Kit.Rand(4 * 5 * 2, 202, 1.4f); if (mode == "nearest")
                    {
                        for (int i = 0; i < 8; i++)
                        {
                            g[i] = (i % 2 == 0 ? -1f : 1f) * (i / 2 * 0.25f); // exact ties
                        }
                    }

                    Run($"grid_{mode}_{pad}_{(ac ? 1 : 0)}", tg => { var xi = tg.InF("x", 1, 2, 5, 6); var gi = tg.InF("g", 1, 4, 5, 2); return ([xi, gi], [tg.B.Emit1("GridSample", [xi, gi], GK.A(("mode", mode), ("padding", pad), ("align_corners", ac)))]); }, x, g);
        }
            }
        }

        foreach (var largest in new[] { true, false })
        {
            var x = Kit.Rand(3 * 10, 203).Select(v => (float)Math.Round(v * 3)).ToArray();
            Run($"topk_{(largest ? 1 : 0)}", tg => { var xi = tg.InF("x", 3, 10); var o = tg.B.Emit("TopK", [xi], GK.A(("k", 4L), ("largest", largest))); return ([xi], [o[0], o[1]]); }, x);
        }
        { var a = Kit.Rand(40, 204, 7f); var b = Kit.Rand(40, 205, 3f); Run("mod_fmod", tg => { var ai = tg.InF("a", 40); var bi = tg.InF("b", 40); return ([ai, bi], [tg.B.Emit1("Mod", [ai, bi], GK.A(("fmod", true)))]); }, a, b); }
        { var a = new float[] { 0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f, 2.4f, 2.6f, 3.5f, -3.5f, 0f, 100.5f }; Run("round", tg => { var ai = tg.InF("a", 12); return ([ai], [tg.B.Emit1("Round", [ai])]); }, a); }
        {
            var x = Kit.Rand(4 * 5 * 6, 206); var w = Kit.Rand(4 * 2 * 9, 207, 0.5f); var b = Kit.Rand(4, 208);
            Save(dir, "convt", "in1", w); Save(dir, "convt", "in2", b);
            Run("convt", tg => { var xi = tg.InF("x", 1, 4, 5, 6); var wi = tg.WF("w", w, 4, 2, 3, 3); var bi = tg.WF("b", b, 4);
                return ([xi], [tg.B.Emit1("ConvTranspose", [xi, wi, bi], GK.A(("groups", 2L), ("strides", new long[] { 2, 2 }), ("pads", new long[] { 1, 1, 1, 1 }), ("output_padding", new long[] { 1, 1 })))]); }, x);
        }
        {
            var d = Kit.Rand(3 * 4 * 5, 209); int[] idx = [0, 2, 1, 3, 3, 0, 1, 2, 2, 1, 0, 3]; // batch_dims=1: [3, 2, 2] -> indexes dims 1,2
            int[] idx2 = [0, 1, 3, 2, 1, 0, 3, 4, 2, 2, 0, 0];
            Run("gathernd_b1", tg => { var di = tg.InF("d", 3, 4, 5); var ii = tg.In("i", DataType.I32, 3, 2, 2); return ([di, ii], [tg.B.Emit1("GatherND", [di, ii], GK.A(("batch_dims", 1L)))]); }, d, idx2);
        }
    }
}

internal static class SpanExt { public static Span<float> Cast(this Span<int> s) => MemoryMarshal.Cast<int, float>(s); }

public class GridSampleTests
{
    [Theory, InlineData("linear"), InlineData("nearest"), InlineData("cubic")]
    public void IdentityGridWithAlignCornersReproducesInput(string mode)
    {
        int N = 2, C = 3, H = 5, W = 6; var x = Kit.Rand(N * C * H * W, 141);
        var grid = new float[N * H * W * 2];
        for (int n = 0; n < N; n++)
        {
            for (int i = 0; i < H; i++)
            {
                for (int j = 0; j < W; j++) { int o = ((n * H + i) * W + j) * 2; grid[o] = -1f + 2f * j / (W - 1); grid[o + 1] = -1f + 2f * i / (H - 1); }
            }
        }

        var tg = new TG(); var xin = tg.InF("x", SK.DSym(C, H, W)); var gin = tg.InF("g", SK.DSym(H, W, 2));
        tg.B.Output("y", tg.B.Emit1("GridSample", [xin, gin], GK.A(("mode", mode), ("align_corners", true), ("padding", "border"))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2); Kit.Set(m, xin, x); Kit.Set(m, gin, grid);
        var p = Kit.P(N); m.Run(p);
        Kit.AssertClose(x, Kit.Out(m, "y", p), 1e-5, 1e-5, mode);
    }

    [Fact]
    public void ZerosPaddingOutsideAndKnownBilinear()
    {
        var x = new float[] { 1, 2, 3, 4 };  // [1,1,2,2]
        var grid = new float[] { 0f, 0f, -3f, 0f, -1f, -1f };  // center (align_corners=false), far outside, top-left corner
        var tg = new TG(); var xin = tg.InF("x", 1, 1, 2, 2); var gin = tg.InF("g", 1, 1, 3, 2);
        tg.B.Output("y", tg.B.Emit1("GridSample", [xin, gin], GK.A(("mode", "linear"), ("padding", "zeros"), ("align_corners", false))));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x); Kit.Set(m, gin, grid);
        var p = Kit.P(1); m.Run(p);
        // center = mean of all four; far outside = 0; (-1,-1) with align_corners=false is at pixel coord (-0.5,-0.5): 1/4 weight on pixel (0,0)
        Kit.AssertClose([2.5f, 0f, 0.25f], Kit.Out(m, "y", p), 1e-6, 1e-6);
    }

    [Fact]
    public void ReportsSupportForGeneralOps()
    {
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(4, 8)); var ix = tg.In("i", DataType.I32, 3);
        tg.B.Output("a", tg.B.Emit1("GatherAxis", [xin, ix], GK.A(("axis", 1L))));
        tg.B.Output("b", tg.B.Emit1("CumSum", [xin], GK.A(("axis", 1L))));
        tg.B.Output("c", tg.B.Emit1("Where", [xin, xin, xin]));
        tg.B.Output("d", tg.B.Emit1("Mod", [xin, xin]));
        foreach (var n in tg.B.Build().Nodes.Where(n => n.Op.Category != OpCategory.View))
        {
            Assert.True(Kit.Device.Supports(n).IsSupported, n.Op.Name);
        }

        tg = new TG(); xin = tg.InF("x", SK.DSym(4, 8)); var bad = tg.InF("i", 3);
        tg.B.Output("a", tg.B.Emit1("GatherAxis", [xin, bad], GK.A(("axis", 1L))));
        var s = Kit.Device.Supports(tg.B.Build().Nodes.Last());
        Assert.False(s.IsSupported); Assert.Contains("indices", s.Reason);
    }
}
