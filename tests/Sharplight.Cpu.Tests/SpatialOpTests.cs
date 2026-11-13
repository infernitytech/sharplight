using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

internal static class SK
{
    public static Dim[] D(params long[] d) => d.Select(v => (Dim)v).ToArray();
    public static Dim[] DSym(params long[] tail) => [TG.T, .. tail.Select(v => (Dim)v)];

    public static byte[] Half(float[] w, out float[] rounded)
    {
        var b = new byte[w.Length * 2]; rounded = new float[w.Length];
        for (int i = 0; i < w.Length; i++) { var h = (Half)w[i]; BitConverter.TryWriteBytes(b.AsSpan(i * 2), h); rounded[i] = (float)h; }
        return b;
    }

    /// <summary>Registers a weight in the requested storage type; returns the value and the f32 values the kernel will effectively see.</summary>
    public static Value Weight(TG tg, string name, float[] w, long[] dims, string type, out float[] effective)
    {
        var shape = new Shape(dims.Select(d => (Dim)d));
        switch (type)
        {
            case "f32": effective = w; return tg.WF(name, w, dims);
            case "f16": { var b = Half(w, out effective); return tg.W(name, DataType.F16, shape, b); }
            case "q8_0": { var q = Quantizers.Q8_0(w); effective = Quantizers.DeqQ8_0(q); return tg.W(name, DataType.Q8_0, shape, q); }
            default: throw new ArgumentException(type);
        }
    }

}

public class ConvTests
{
    public record Case(string Name, int C, int H, int W, int O, int KH, int KW, int[] Stride, int[] Pad, int[] Dil, int Groups, bool Bias);

    private static readonly Case[] Cases =
    [
        new("3x3-s1-p1-bias", 5, 11, 13, 7, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, true),
        new("3x3-s1-p1-nobias", 5, 11, 13, 7, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, false),
        new("1x1-tail", 16, 7, 9, 20, 1, 1, [1, 1], [0, 0, 0, 0], [1, 1], 1, true),             // HW=63: direct path + partial tile
        new("1x1-exact-tiles", 16, 8, 8, 24, 1, 1, [1, 1], [0, 0, 0, 0], [1, 1], 1, false),     // HW=64: direct, no tail
        new("1x1-s2", 12, 9, 9, 10, 1, 1, [2, 2], [0, 0, 0, 0], [1, 1], 1, true),
        new("1x1-hw1", 33, 1, 1, 17, 1, 1, [1, 1], [0, 0, 0, 0], [1, 1], 1, true),
        new("3x3-s2-p1", 6, 14, 14, 9, 3, 3, [2, 2], [1, 1, 1, 1], [1, 1], 1, true),
        new("7x7-s2-p3", 3, 33, 31, 16, 7, 7, [2, 2], [3, 3, 3, 3], [1, 1], 1, true),
        new("asym-pads", 4, 10, 12, 6, 3, 3, [1, 1], [1, 2, 0, 3], [1, 1], 1, true),
        new("dil2", 4, 15, 14, 6, 3, 3, [1, 1], [2, 2, 2, 2], [2, 2], 1, true),
        new("dil2-3-s2", 3, 17, 19, 5, 3, 3, [2, 1], [2, 3, 2, 3], [2, 3], 1, false),
        new("rect-kernel", 4, 12, 15, 5, 3, 5, [1, 2], [1, 2, 1, 2], [1, 1], 1, true),
        new("long-K", 40, 9, 10, 11, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, true),             // K=360 > KC
        new("long-K-1x1", 300, 6, 7, 9, 1, 1, [1, 1], [0, 0, 0, 0], [1, 1], 1, true),          // K=300 > KC, direct
        new("groups2", 6, 10, 11, 8, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 2, true),
        new("groups4-s2", 8, 13, 13, 12, 3, 3, [2, 2], [1, 1, 1, 1], [1, 1], 4, false),
        new("depthwise-3x3", 9, 14, 15, 9, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 9, true),
        new("depthwise-3x3-s2", 9, 14, 15, 9, 3, 3, [2, 2], [1, 1, 1, 1], [1, 1], 9, false),
        new("depthwise-5x5-dil2", 4, 17, 18, 4, 5, 5, [1, 1], [4, 4, 4, 4], [2, 2], 4, true),
        new("depthwise-mult2", 4, 9, 10, 8, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 4, true),
        new("q-3x3", 4, 10, 11, 8, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, true),                // 288 weights: whole q8_0 blocks
        new("q-groups2", 8, 9, 9, 8, 3, 3, [2, 2], [1, 1, 1, 1], [1, 1], 2, true),
        new("q-depthwise-5x5", 32, 11, 12, 32, 5, 5, [1, 1], [2, 2, 2, 2], [1, 1], 32, true),
        new("wide", 8, 70, 70, 20, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, true),
    ];

    public static IEnumerable<object[]> Params()
    {
        foreach (var c in Cases)
        {
            foreach (var n in new[] { 1, 3 })
            {
                yield return [c.Name, n, "f32"];
            }
        }

        foreach (var name in new[] { "3x3-s1-p1-bias", "groups2", "depthwise-3x3", "1x1-tail", "q-3x3", "q-groups2", "q-depthwise-5x5" })
        {
            foreach (var t in new[] { "f16", "q8_0" })
            {
                if (t != "q8_0" || name == "1x1-tail" || name.StartsWith("q-"))
                {
                    yield return [name, 3, t];
                }
            }
        }
    }

    [Theory, MemberData(nameof(Params))]
    public void Conv(string name, int N, string wtype)
    {
        var c = Cases.Single(k => k.Name == name);
        int Cg = c.C / c.Groups;
        long wn = (long)c.O * Cg * c.KH * c.KW;
        // q8_0 needs a multiple of 32 weight elements
        var w = Kit.Rand((int)wn, 11, 0.5f);
        if (wtype == "q8_0")
        {
            Assert.True(wn % 32 == 0, "q8_0 case needs whole blocks");
        }

        var x = Kit.Rand(N * c.C * c.H * c.W, 12);
        var bias = c.Bias ? Kit.Rand(c.O, 13) : null;
        var tg = new TG();
        var xin = tg.InF("x", SK.DSym(c.C, c.H, c.W));
        var wv = SK.Weight(tg, "w", w, [c.O, Cg, c.KH, c.KW], wtype, out var weff);
        var ins = new List<Value> { xin, wv };
        if (bias != null)
        {
            ins.Add(tg.WF("b", bias, c.O));
        }

        var attrs = Attributes.From(("groups", (long)c.Groups), ("strides", c.Stride.Select(v => (long)v).ToArray()),
            ("pads", c.Pad.Select(v => (long)v).ToArray()), ("dilations", c.Dil.Select(v => (long)v).ToArray()));
        tg.B.Output("y", tg.B.Emit1("Conv", ins, attrs));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        Kit.Set(m, xin, x);
        var p = Kit.P(N); m.Run(p);
        var exp = SRef.Conv(x, N, c.C, c.H, c.W, weff, c.O, c.KH, c.KW, bias, c.Stride, c.Pad, c.Dil, c.Groups, out _, out _);
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4, $"{name} N={N} {wtype}");
    }

    [Fact]
    public void ConvOnStridedAndChainedInputs()
    {
        // NHWC -> NCHW permute feeding a conv, then conv -> relu -> conv (activation weights stay in the arena)
        int N = 2, H = 9, W = 10, C = 5, O = 6;
        var x = Kit.Rand(N * H * W * C, 21);
        var w1 = Kit.Rand(O * C * 9, 22, 0.4f); var w2 = Kit.Rand(O * O, 23, 0.4f);
        var tg = new TG();
        var xin = tg.InF("x", TG.T, H, W, C);
        var nchw = tg.B.Permute(xin, 0, 3, 1, 2);
        var w1v = tg.WF("w1", w1, O, C, 3, 3); var w2v = tg.WF("w2", w2, O, O, 1, 1);
        var a1 = Attributes.From(("pads", new long[] { 1, 1, 1, 1 }));
        var y1 = tg.B.Emit1("Conv", [nchw, w1v], a1);
        var y2 = tg.B.Emit1("Conv", [tg.B.Unary("Relu", y1), w2v]);
        tg.B.Output("y", y2);
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2);
        Kit.Set(m, xin, x);
        var p = Kit.P(N); m.Run(p);
        var xc = new float[N * C * H * W];
        for (int n = 0; n < N; n++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int ww = 0; ww < W; ww++)
                {
                    for (int c = 0; c < C; c++)
                    {
                        xc[((n * C + c) * H + h) * W + ww] = x[((n * H + h) * W + ww) * C + c];
                    }
                }
            }
        }

        var r1 = SRef.Conv(xc, N, C, H, W, w1, O, 3, 3, null, [1, 1], [1, 1, 1, 1], [1, 1], 1, out int h1, out int w1o);
        r1 = r1.Select(v => Math.Max(0, v)).ToArray();
        var r2 = SRef.Conv(r1, N, O, h1, w1o, w2, O, 1, 1, null, [1, 1], [0, 0, 0, 0], [1, 1], 1, out _, out _);
        Kit.AssertClose(r2, Kit.Out(m, "y", p), 1e-4, 1e-4);
    }

    [Fact]
    public void SupportsReasons()
    {
        var tg = new TG();
        var xin = tg.InF("x", SK.DSym(4, 8, 8));
        var xin2 = tg.InF("w", SK.D(4, 4, 3, 3)); // activation as weight
        tg.B.Output("y", tg.B.Emit1("Conv", [xin, xin2]));
        var g = tg.B.Build();
        var node = g.Nodes.First(n => n.Op.Name == "Conv");
        var s = Kit.Device.Supports(node);
        Assert.False(s.IsSupported);
        Assert.Contains("weight", s.Reason);
    }
}

public class PoolTests
{
    public static IEnumerable<object[]> Cases()
    {
        // name, mode, H, W, kernel, stride, pad(4), dil, ceil, includePad
        object[][] c =
        [
            ["max-3x3-s2-p1", "max", 14, 15, new[] { 3, 3 }, new[] { 2, 2 }, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, false, false],
            ["max-2x2-s2", "max", 8, 9, new[] { 2, 2 }, new[] { 2, 2 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, false, false],
            ["max-ceil-odd", "max", 7, 9, new[] { 2, 2 }, new[] { 2, 2 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, true, false],
            ["max-ceil-pad-drop", "max", 6, 6, new[] { 3, 3 }, new[] { 2, 2 }, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, true, false],
            ["max-ceil-s3", "max", 10, 11, new[] { 3, 3 }, new[] { 3, 3 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, true, false],
            ["max-dil2", "max", 13, 12, new[] { 3, 3 }, new[] { 1, 1 }, new[] { 2, 2, 2, 2 }, new[] { 2, 2 }, false, false],
            ["max-dil-ceil-asym", "max", 13, 12, new[] { 2, 3 }, new[] { 2, 2 }, new[] { 1, 0, 1, 2 }, new[] { 2, 1 }, true, false],
            ["avg-3x3-s2-p1-excl", "avg", 14, 15, new[] { 3, 3 }, new[] { 2, 2 }, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, false, false],
            ["avg-3x3-s2-p1-incl", "avg", 14, 15, new[] { 3, 3 }, new[] { 2, 2 }, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, false, true],
            ["avg-ceil-excl", "avg", 7, 9, new[] { 2, 2 }, new[] { 2, 2 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, true, false],
            ["avg-ceil-incl-pad", "avg", 6, 6, new[] { 3, 3 }, new[] { 2, 2 }, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, true, true],
            ["avg-ceil-incl-nopad", "avg", 8, 8, new[] { 3, 3 }, new[] { 2, 2 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, true, true],
            ["avg-7x7", "avg", 7, 7, new[] { 7, 7 }, new[] { 1, 1 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, false, false],
        ];
        foreach (var o in c)
        {
            foreach (var n in new[] { 1, 3 })
            {
                yield return [.. o, n];
            }
        }
    }

    [Theory, MemberData(nameof(Cases))]
    public void Pool(string name, string mode, int H, int W, int[] k, int[] s, int[] pad, int[] dil, bool ceil, bool includePad, int N)
    {
        int C = 4;
        var x = Kit.Rand(N * C * H * W, 31);
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(C, H, W));
        var attrs = Attributes.From(("mode", mode), ("kernel", k.Select(v => (long)v).ToArray()), ("strides", s.Select(v => (long)v).ToArray()),
            ("pads", pad.Select(v => (long)v).ToArray()), ("dilations", dil.Select(v => (long)v).ToArray()), ("ceil_mode", ceil), ("count_include_pad", includePad));
        tg.B.Output("y", tg.B.Emit1("Pool", [xin], attrs));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        Kit.Set(m, xin, x);
        var p = Kit.P(N); m.Run(p);
        var exp = SRef.Pool(x, N, C, H, W, mode == "max", k, s, pad, dil, ceil, includePad, out int ho, out int wo);
        var shape = m.Graph.Outputs["y"].Shape;
        Assert.Equal(ho, (int)shape[2].Fixed); Assert.Equal(wo, (int)shape[3].Fixed);
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-5, 1e-5, name);
    }
}

public class ReduceTests
{
    public static IEnumerable<object[]> Cases()
    {
        long[][] shapes = [[2, 3, 4, 5], [3, 16, 20, 17], [4, 9, 33], [37], [5, 40]];
        foreach (var mode in new[] { "sum", "mean", "max", "min" })
        {
            foreach (var (shape, axes) in new (long[], long[])[]
            {
                ([2, 3, 4, 5], [1]), ([2, 3, 4, 5], [0, 2]), ([2, 3, 4, 5], [2, 3]), ([2, 3, 4, 5], [-1]), ([2, 3, 4, 5], [0, 1, 2, 3]), ([2, 3, 4, 5], [1, 3]),
                ([3, 16, 20, 17], [2, 3]), ([3, 16, 20, 17], [0, 1]), ([3, 16, 20, 17], [1]), ([3, 16, 20, 17], [3]), ([3, 16, 20, 17], [0, 3]), ([3, 16, 20, 17], [0, 1, 2]),
                ([4, 9, 33], [2]), ([4, 9, 33], [0]), ([4, 9, 33], [1, 2]), ([37], [0]), ([5, 40], [1]), ([5, 40], [0]),
            })
            {
                foreach (var keep in new[] { true, false })
                {
                    yield return [mode, shape, axes, keep, false];
                }
            }
        }
        // strided (permuted) input: NHWC memory seen as NCHW
        foreach (var mode in new[] { "sum", "mean", "max", "min" })
        {
            foreach (var axes in new long[][] { [2, 3], [1], [0, 1], [1, 2, 3] })
            {
                yield return [mode, new long[] { 2, 24, 7, 9 }, axes, true, true];
            }
        }
    }

    [Theory, MemberData(nameof(Cases))]
    public void Reduce(string mode, long[] dims, long[] axes, bool keep, bool permuted)
    {
        // 'permuted': logical [N,C,H,W] = Permute(physical [N,H,W,C]); axes refer to the logical layout
        var tg = new TG();
        long total = SRef.Prod(dims);
        var x = Kit.Rand((int)total, 41);
        Value xin, src;
        float[] logical = x;
        if (permuted)
        {
            long N = dims[0], C = dims[1], H = dims[2], W = dims[3];
            xin = tg.InF("x", SK.D(N, H, W, C));
            src = tg.B.Permute(xin, 0, 3, 1, 2);
            logical = new float[total];
            for (long n = 0; n < N; n++)
            {
                for (long h = 0; h < H; h++)
                {
                    for (long w = 0; w < W; w++)
                    {
                        for (long c = 0; c < C; c++)
                        {
                            logical[((n * C + c) * H + h) * W + w] = x[((n * H + h) * W + w) * C + c];
                        }
                    }
                }
            }
        }
        else { xin = tg.InF("x", SK.D(dims)); src = xin; }
        tg.B.Output("y", tg.B.Emit1("Reduce", [src], Attributes.From(("mode", mode), ("axes", axes), ("keepdims", keep))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(1); m.Run(p);
        var exp = SRef.Reduce(logical, dims, axes, mode, keep, out var od);
        var shape = m.Graph.Outputs["y"].Shape;
        Assert.Equal(od, shape.Dims.Select(d => d.Fixed).ToArray());
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-5, $"{mode} {string.Join(",", dims)} axes {string.Join(",", axes)}");
    }

    [Fact]
    public void ReduceWithSymbolicBatch()
    {
        int N = 3, C = 24, H = 7, W = 7;
        var x = Kit.Rand(N * C * H * W, 42);
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(C, H, W));
        tg.B.Output("gap", tg.B.Emit1("Reduce", [xin], Attributes.From(("mode", "mean"), ("axes", new long[] { 2, 3 }), ("keepdims", false))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        Kit.Set(m, xin, x);
        var p = Kit.P(N); m.Run(p);
        Kit.AssertClose(SRef.Reduce(x, [N, C, H, W], [2, 3], "mean", false, out _), Kit.Out(m, "gap", p), 1e-5, 1e-5);
    }
}

public class BatchMatMulTests
{
    public static IEnumerable<object[]> Cases()
    {
        (long[] a, long[] b, bool ta, bool tb, float alpha)[] c =
        [
            ([2, 3, 5, 7], [2, 3, 7, 4], false, false, 1f),
            ([2, 3, 5, 7], [2, 3, 4, 7], false, true, 0.125f),
            ([2, 3, 7, 5], [2, 3, 7, 4], true, false, 1f),
            ([2, 3, 7, 5], [2, 3, 4, 7], true, true, 2f),
            ([4, 1, 6, 9], [1, 3, 9, 5], false, false, 1f),
            ([3, 6, 9], [9, 5], false, false, 1f),
            ([6, 9], [3, 9, 5], false, false, 1f),
            ([4, 1, 6, 9], [3, 5, 9], false, true, 1f),
            ([2, 5, 7], [2, 7, 3], false, false, 1f),
            ([1, 70, 300], [1, 300, 75], false, false, 1f),     // crosses KC and tile edges
            ([2, 70, 300], [2, 75, 300], false, true, 0.5f),
            ([5, 33], [33, 40], false, false, 1f),
            ([1, 1, 1, 64], [1, 1, 64, 130], false, false, 1f),  // M=1 (decode-like)
            ([2, 12, 197, 64], [2, 12, 64, 197], false, false, 0.125f), // attention-like
        ];
        foreach (var t in c)
        {
            yield return [t.a, t.b, t.ta, t.tb, t.alpha];
        }
    }

    [Theory, MemberData(nameof(Cases))]
    public void Bmm(long[] ad, long[] bd, bool ta, bool tb, float alpha)
    {
        var a = Kit.Rand((int)SRef.Prod(ad), 51); var b = Kit.Rand((int)SRef.Prod(bd), 52);
        var tg = new TG(); var ain = tg.InF("a", SK.D(ad)); var bin = tg.InF("b", SK.D(bd));
        tg.B.Output("y", tg.B.Emit1("BatchMatMul", [ain, bin], Attributes.From(("trans_a", ta), ("trans_b", tb), ("alpha", alpha))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, ain, a); Kit.Set(m, bin, b);
        var p = Kit.P(1); m.Run(p);
        var exp = SRef.Bmm(a, ad, b, bd, ta, tb, alpha, out var od);
        Assert.Equal(od, m.Graph.Outputs["y"].Shape.Dims.Select(d => d.Fixed).ToArray());
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4);
    }

    /// <summary>Attention-style on permuted (strided) views: q [B,T,H,D] -> [B,H,T,D]; k [B,S,H,D] -> [B,H,D,S].</summary>
    [Fact]
    public void AttentionOnStridedViews()
    {
        int B = 2, T = 13, S = 21, H = 3, D = 17;
        var q = Kit.Rand(B * T * H * D, 53); var k = Kit.Rand(B * S * H * D, 54);
        var tg = new TG(); var qin = tg.InF("q", SK.D(B, T, H, D)); var kin = tg.InF("k", SK.D(B, S, H, D));
        var qp = tg.B.Permute(qin, 0, 2, 1, 3); var kp = tg.B.Permute(kin, 0, 2, 3, 1);
        tg.B.Output("s", tg.B.Emit1("BatchMatMul", [qp, kp], Attributes.From(("alpha", 0.25f))));
        // transposed-flag variant of the same product: k permuted to [B,H,S,D] with trans_b
        var kp2 = tg.B.Permute(kin, 0, 2, 1, 3);
        tg.B.Output("s2", tg.B.Emit1("BatchMatMul", [qp, kp2], Attributes.From(("trans_b", true), ("alpha", 0.25f))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, qin, q); Kit.Set(m, kin, k);
        var p = Kit.P(1); m.Run(p);
        var ql = new float[B * H * T * D]; var kl = new float[B * H * D * S];
        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int t = 0; t < T; t++)
                {
                    for (int d = 0; d < D; d++)
                    {
                        ql[((b * H + h) * T + t) * D + d] = q[((b * T + t) * H + h) * D + d];
                    }
                }
            }
        }

        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int s = 0; s < S; s++)
                {
                    for (int d = 0; d < D; d++)
                    {
                        kl[((b * H + h) * D + d) * S + s] = k[((b * S + s) * H + h) * D + d];
                    }
                }
            }
        }

        var exp = SRef.Bmm(ql, [B, H, T, D], kl, [B, H, D, S], false, false, 0.25f, out _);
        Kit.AssertClose(exp, Kit.Out(m, "s", p), 1e-4, 1e-4);
        Kit.AssertClose(exp, Kit.Out(m, "s2", p), 1e-4, 1e-4);
    }

    [Fact]
    public void SymbolicM()
    {
        int T = 5, K = 19, N = 23;
        var a = Kit.Rand(T * K, 55); var b = Kit.Rand(K * N, 56);
        var tg = new TG(); var ain = tg.InF("a", SK.DSym(K)); var bin = tg.InF("b", SK.D(K, N));
        tg.B.Output("y", tg.B.Emit1("BatchMatMul", [ain, bin]));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 8);
        Kit.Set(m, ain, a); Kit.Set(m, bin, b);
        var p = Kit.P(T); m.Run(p);
        Kit.AssertClose(SRef.Bmm(a, [T, K], b, [K, N], false, false, 1f, out _), Kit.Out(m, "y", p), 1e-4, 1e-4);
    }
}

public class PadResizeTests
{
    public static IEnumerable<object[]> PadCases()
    {
        foreach (var mode in new[] { "constant", "reflect", "edge" })
        {
            foreach (var (dims, pads) in new (long[], long[])[]
            {
                ([2, 3, 5, 6], [0, 0, 1, 2, 0, 0, 3, 1]),
                ([2, 3, 5, 6], [0, 1, 2, 2, 1, 0, 1, 4]),
                ([2, 3, 5, 6], [0, 0, -1, 2, 0, 0, 2, -2]),     // negative pads crop
                ([2, 3, 5, 6], [0, 0, -2, -1, 0, 0, -1, -3]),
                ([4, 7], [2, 3, 1, 1]),
                ([9], [4, 3]),
                ([3, 4, 5], [1, 0, 2, 2, 1, 1]),
                ([1, 2, 6, 40], [0, 0, 0, 5, 0, 0, 0, 5]),
            })
            {
                yield return [mode, dims, pads, false];
            }
        }

        yield return ["reflect", new long[] { 2, 3, 5, 6 }, new long[] { 0, 0, 1, 2, 0, 0, 3, 1 }, true];
        yield return ["constant", new long[] { 2, 3, 5, 6 }, new long[] { 0, 0, 1, 2, 0, 0, 3, 1 }, true];
    }

    [Theory, MemberData(nameof(PadCases))]
    public void Pad(string mode, long[] dims, long[] pads, bool permuted)
    {
        var x = Kit.Rand((int)SRef.Prod(dims), 61);
        var tg = new TG();
        Value xin, src; float[] logical = x;
        if (permuted)
        {
            long N = dims[0], C = dims[1], H = dims[2], W = dims[3];
            xin = tg.InF("x", SK.D(N, H, W, C)); src = tg.B.Permute(xin, 0, 3, 1, 2);
            logical = new float[x.Length];
            for (long n = 0; n < N; n++)
            {
                for (long h = 0; h < H; h++)
                {
                    for (long w = 0; w < W; w++)
                    {
                        for (long c = 0; c < C; c++)
                        {
                            logical[((n * C + c) * H + h) * W + w] = x[((n * H + h) * W + w) * C + c];
                        }
                    }
                }
            }
        }
        else { xin = tg.InF("x", SK.D(dims)); src = xin; }
        tg.B.Output("y", tg.B.Emit1("Pad", [src], Attributes.From(("pads", pads), ("mode", mode), ("value", 1.5f))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(1); m.Run(p);
        var exp = SRef.Pad(logical, dims, pads, mode, 1.5f, out var od);
        Assert.Equal(od, m.Graph.Outputs["y"].Shape.Dims.Select(d => d.Fixed).ToArray());
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 0, 0, $"{mode} pads {string.Join(",", pads)}");
    }

    [Fact]
    public void PadSymbolicBatch()
    {
        int N = 3; var x = Kit.Rand(N * 2 * 4 * 5, 62);
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(2, 4, 5));
        tg.B.Output("y", tg.B.Emit1("Pad", [xin], Attributes.From(("pads", new long[] { 0, 0, 1, 2, 0, 0, 1, 1 }), ("mode", "edge"))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        Kit.Set(m, xin, x);
        var p = Kit.P(N); m.Run(p);
        Kit.AssertClose(SRef.Pad(x, [N, 2, 4, 5], [0, 0, 1, 2, 0, 0, 1, 1], "edge", 0, out _), Kit.Out(m, "y", p), 0, 0);
    }

    public static IEnumerable<object[]> ResizeCases()
    {
        foreach (var coord in new[] { "half_pixel", "align_corners", "asymmetric", "pytorch_half_pixel" })
        {
            foreach (var nm in new[] { "round_prefer_floor", "round_prefer_ceil", "floor", "ceil" })
            {
                foreach (var (oh, ow) in new[] { (8, 10), (7, 9), (3, 2), (2, 7), (4, 5), (1, 1), (12, 15) })
                {
                    yield return ["nearest", coord, nm, oh, ow];
                }
            }

            foreach (var (oh, ow) in new[] { (8, 10), (7, 9), (3, 2), (2, 7), (4, 5), (1, 1), (12, 15) })
            {
                yield return ["linear", coord, "round_prefer_floor", oh, ow];
            }
        }
    }

    [Theory, MemberData(nameof(ResizeCases))]
    public void Resize(string mode, string coord, string nearest, int oh, int ow)
    {
        int N = 2, C = 3, H = 4, W = 5;
        var x = Kit.Rand(N * C * H * W, 63);
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(C, H, W));
        tg.B.Output("y", tg.B.Emit1("Resize", [xin], Attributes.From(("size", new long[] { oh, ow }), ("mode", mode), ("coord", coord), ("nearest_mode", nearest))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2);
        Kit.Set(m, xin, x);
        var p = Kit.P(N); m.Run(p);
        var exp = SRef.Resize(x, N, C, H, W, oh, ow, mode, coord, nearest);
        Kit.AssertClose(exp, Kit.Out(m, "y", p), mode == "nearest" ? 0 : 1e-6, 1e-6, $"{mode} {coord} {nearest} {oh}x{ow}");
    }
}

public class NewElementwiseTests
{
    private static float[] Run1(string op, float[] x, Attributes? attrs = null)
    {
        var tg = new TG(); var xin = tg.InF("x", TG.T, x.Length / 3);
        tg.B.Output("y", tg.B.Emit1(op, [xin], attrs));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(3); m.Run(p);
        return Kit.Out(m, "y", p);
    }

    [Fact]
    public void ErfIsAccurate()
    {
        int n = 3 * 4001; var x = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = -8f + 16f * i / (n - 1);
        }

        var y = Run1("Erf", x);
        double worst = 0; float wx = 0;
        for (int i = 0; i < n; i++) { double e = Math.Abs(y[i] - Math2.Erf(x[i])); if (e > worst) { worst = e; wx = x[i]; } }
        Assert.True(worst < 2e-7, $"max abs erf error {worst:E3} at x={wx}");
    }

    [Fact]
    public void ErfOddLengthTailAndSpecials()
    {
        var x = new float[] { 0f, 1e-8f, -1e-3f, 0.5f, -0.5f, 3.5f, -3.5f, 10f, -10f };
        var y = Run1("Erf", x);
        for (int i = 0; i < x.Length; i++)
        {
            Assert.True(Math.Abs(y[i] - Math2.Erf(x[i])) < 2e-7, $"erf({x[i]}) = {y[i]}");
        }
    }

    [Fact]
    public void HardSwishAndSigmoid()
    {
        var x = Kit.Rand(3 * 37, 71, 8f);
        Kit.AssertClose(x.Select(v => (float)(v * Math.Clamp(v / 6.0 + 0.5, 0, 1))).ToArray(), Run1("HardSwish", x), 1e-6, 1e-6);
        Kit.AssertClose(x.Select(v => (float)Math.Clamp(0.2 * v + 0.5, 0, 1)).ToArray(), Run1("HardSigmoid", x), 1e-6, 1e-6);
        Kit.AssertClose(x.Select(v => (float)Math.Clamp(Math.Round(1.0 / 6, 10) * v + 0.5, 0, 1)).ToArray(),
            Run1("HardSigmoid", x, Attributes.From(("alpha", 1f / 6f), ("beta", 0.5f))), 1e-6, 1e-6);
        Kit.AssertClose(x.Select(v => (float)Math.Clamp(0.1 * v - 0.3, 0, 1)).ToArray(),
            Run1("HardSigmoid", x, Attributes.From(("alpha", 0.1f), ("beta", -0.3f))), 1e-6, 1e-6);
    }

    [Fact]
    public void ClipVariants()
    {
        var x = Kit.Rand(3 * 37, 72, 4f);
        Kit.AssertClose(x.Select(v => Math.Min(Math.Max(v, -1f), 2.5f)).ToArray(), Run1("Clip", x, Attributes.From(("min", -1f), ("max", 2.5f))), 0, 0);
        Kit.AssertClose(x.Select(v => Math.Max(v, 0f)).ToArray(), Run1("Clip", x, Attributes.From(("min", 0f))), 0, 0);
        Kit.AssertClose(x.Select(v => Math.Min(v, 6f)).ToArray(), Run1("Clip", x, Attributes.From(("max", 6f))), 0, 0);
        Kit.AssertClose(x.Select(v => Math.Clamp(v, 0f, 6f)).ToArray(), Run1("Clip", x, Attributes.From(("min", 0f), ("max", 6f))), 0, 0);
    }

    [Theory]
    [InlineData(2f), InlineData(0.5f), InlineData(3f), InlineData(1f), InlineData(-1.5f), InlineData(0f)]
    public void PowScalarExponent(float e)
    {
        var x = Kit.Rand(3 * 37, 73).Select(v => Math.Abs(v) + 0.1f).ToArray();
        var tg = new TG(); var xin = tg.InF("x", TG.T, 37);
        tg.B.Output("y", tg.B.Binary("Pow", xin, tg.WF("e", [e], 1)));
        using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, x);
        var p = Kit.P(3); m.Run(p);
        Kit.AssertClose(x.Select(v => (float)Math.Pow(v, e)).ToArray(), Kit.Out(m, "y", p), 1e-6, 1e-5);
    }

    [Fact]
    public void Nchw4dBroadcastConstants()
    {
        int N = 3, C = 6, H = 5, W = 7;
        var x = Kit.Rand(N * C * H * W, 74); var sc = Kit.Rand(C, 75); var bi = Kit.Rand(C, 76);
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(C, H, W));
        var scv = tg.WF("sc", sc, C, 1, 1); var biv = tg.WF("bi", bi, C, 1, 1);
        var y = tg.B.Add(tg.B.Mul(xin, scv), biv);
        tg.B.Output("y", y);
        tg.B.Output("pw", tg.B.Binary("Pow", tg.B.Unary("Abs", xin), tg.WF("pe", [1.7f], 1, 1, 1)));
        tg.B.Output("sub", tg.B.Sub(xin, tg.WF("c4", Kit.Rand(C, 77), 1, C, 1, 1)));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3); Kit.Set(m, xin, x);
        var p = Kit.P(N); m.Run(p);
        var exp = new float[x.Length]; var pw = new float[x.Length];
        for (int i = 0; i < x.Length; i++) { int c = i / (H * W) % C; exp[i] = x[i] * sc[c] + bi[c]; pw[i] = (float)Math.Pow(Math.Abs(x[i]), 1.7f); }
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-6, 1e-5);
        Kit.AssertClose(pw, Kit.Out(m, "pw", p), 1e-6, 1e-5);
    }

    [Fact]
    public void SoftmaxAndLayerNormOnHigherRanks()
    {
        int N = 3;
        // 4-D softmax over the last axis, 3-D layer norm with weight/bias
        var x4 = Kit.Rand(N * 2 * 3 * 17, 78); var x3 = Kit.Rand(N * 5 * 24, 79);
        var w = Kit.Rand(24, 80); var b = Kit.Rand(24, 81);
        var tg = new TG(); var i4 = tg.InF("x4", SK.DSym(2, 3, 17)); var i3 = tg.InF("x3", SK.DSym(5, 24));
        tg.B.Output("sm", tg.B.Emit1("Softmax", [i4]));
        tg.B.Output("ln", tg.B.Emit1("LayerNorm", [i3, tg.WF("w", w, 24), tg.WF("b", b, 24)], Attributes.From(("eps", 1e-5f))));
        tg.B.Output("rms", tg.B.RmsNorm(tg.B.Permute(i4, 0, 1, 3, 2), null, 1e-6f));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3); Kit.Set(m, i4, x4); Kit.Set(m, i3, x3);
        var p = Kit.P(N); m.Run(p);
        Kit.AssertClose(Ref.Softmax(x4, N * 2 * 3, 17, 1f), Kit.Out(m, "sm", p), 1e-6, 1e-5);
        var ln = new float[x3.Length];
        for (int r = 0; r < N * 5; r++)
        {
            double mean = 0; for (int i = 0; i < 24; i++)
            {
                mean += x3[r * 24 + i];
            }

            mean /= 24;
            double v = 0; for (int i = 0; i < 24; i++)
            {
                v += Math.Pow(x3[r * 24 + i] - mean, 2);
            }

            v /= 24;
            for (int i = 0; i < 24; i++)
            {
                ln[r * 24 + i] = (float)((x3[r * 24 + i] - mean) / Math.Sqrt(v + 1e-5) * w[i] + b[i]);
            }
        }
        Kit.AssertClose(ln, Kit.Out(m, "ln", p), 1e-5, 1e-4);
        var perm = new float[x4.Length];
        for (int r = 0; r < N * 2; r++)
        {
            for (int a = 0; a < 3; a++)
            {
                for (int c = 0; c < 17; c++)
                {
                    perm[(r * 17 + c) * 3 + a] = x4[(r * 3 + a) * 17 + c];
                }
            }
        }

        Kit.AssertClose(Ref.RmsNorm(perm, N * 2 * 17, 3, null, 1e-6f), Kit.Out(m, "rms", p), 1e-5, 1e-4);
    }

    [Fact]
    public void GeluUsesAccurateErf()
    {
        var x = Kit.Rand(3 * 101, 82, 5f);
        var y = Run1("Gelu", x.Concat(new float[3 * 101 - x.Length]).ToArray());
        for (int i = 0; i < x.Length; i++)
        {
            Assert.True(Math.Abs(y[i] - 0.5 * x[i] * (1 + Math2.Erf(x[i] / Math.Sqrt(2)))) < 2e-6);
        }
    }
}

public class SupportTests
{
    private static Node NodeOf(TG tg) => tg.B.Build().Nodes.Last();

    [Fact]
    public void ReportsReasonsForUnsupportedAttributeCombos()
    {
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(4, 8, 8));
        tg.B.Output("y", tg.B.Emit1("Pool", [xin], Attributes.From(("mode", "avg"), ("kernel", new long[] { 2, 2 }), ("dilations", new long[] { 2, 2 }))));
        var s = Kit.Device.Supports(NodeOf(tg));
        Assert.False(s.IsSupported); Assert.Contains("dilation", s.Reason);

        tg = new TG(); xin = tg.InF("x", SK.DSym(4, 8, 8));
        tg.B.Output("y", tg.B.Emit1("Resize", [xin], Attributes.From(("size", new long[] { 4, 4 }), ("mode", "cubic"))));
        s = Kit.Device.Supports(NodeOf(tg));
        Assert.False(s.IsSupported); Assert.Contains("cubic", s.Reason);

        tg = new TG(); xin = tg.InF("x", SK.DSym(4, 8, 8));
        tg.B.Output("y", tg.B.Emit1("Pad", [xin], Attributes.From(("pads", new long[] { 0, 0, 0, 0, 0, 0, 0, 0 }), ("mode", "wrap"))));
        s = Kit.Device.Supports(NodeOf(tg));
        Assert.False(s.IsSupported); Assert.Contains("wrap", s.Reason);
    }

    [Fact]
    public void ReportsSupportForNewOps()
    {
        var tg = new TG(); var xin = tg.InF("x", SK.DSym(4, 8, 8));
        tg.B.Output("a", tg.B.Emit1("Pool", [xin], Attributes.From(("mode", "max"), ("kernel", new long[] { 2, 2 }))));
        tg.B.Output("b", tg.B.Emit1("Reduce", [xin], Attributes.From(("mode", "sum"), ("axes", new long[] { 2 }))));
        tg.B.Output("c", tg.B.Emit1("Clip", [xin], Attributes.From(("min", 0f))));
        tg.B.Output("d", tg.B.Emit1("Erf", [xin]));
        tg.B.Output("e", tg.B.Binary("Pow", xin, xin));
        tg.B.Output("f", tg.B.Emit1("BatchMatMul", [xin, tg.B.Permute(xin, 0, 1, 3, 2)]));
        foreach (var n in tg.B.Build().Nodes.Where(n => n.Op.Category != OpCategory.View))
        {
            Assert.True(Kit.Device.Supports(n).IsSupported, n.Op.Name + ": " + Kit.Device.Supports(n).Reason);
        }
    }
}

/// <summary>Writes Resize results to $SHARPLIGHT_ORT_DUMP (used by .work/venv onnxruntime cross-check scripts); no-op otherwise.</summary>
public class OrtDumpTests
{
    [Fact]
    public void DumpResize()
    {
        var dir = Environment.GetEnvironmentVariable("SHARPLIGHT_ORT_DUMP");
        if (string.IsNullOrEmpty(dir))
        {
            return;
        }

        Directory.CreateDirectory(dir);
        int N = 1, C = 2, H = 4, W = 5;
        var x = Kit.Rand(N * C * H * W, 63);
        File.WriteAllBytes(Path.Combine(dir, "x.bin"), MemoryMarshal.AsBytes(x.AsSpan()).ToArray());
        foreach (var o in PadResizeTests.ResizeCases())
        {
            string mode = (string)o[0], coord = (string)o[1], nm = (string)o[2]; int oh = (int)o[3], ow = (int)o[4];
            var tg = new TG(); var xin = tg.InF("x", SK.D(N, C, H, W));
            tg.B.Output("y", tg.B.Emit1("Resize", [xin], Attributes.From(("size", new long[] { oh, ow }), ("mode", mode), ("coord", coord), ("nearest_mode", nm))));
            using var m = Kit.Compile(tg.B.Build());
            Kit.Set(m, xin, x);
            var p = Kit.P(1); m.Run(p);
            File.WriteAllBytes(Path.Combine(dir, $"{mode}_{coord}_{nm}_{oh}x{ow}.bin"), MemoryMarshal.AsBytes(Kit.Out(m, "y", p).AsSpan()).ToArray());
        }
    }
}
