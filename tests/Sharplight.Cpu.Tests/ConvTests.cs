using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

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
        new("long-K-packed-B", 160, 9, 9, 8, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, true),     // K=1440: im2col panels, not tile-packed
        new("depthwise-wide-s1", 6, 13, 53, 6, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 6, true),   // vector interior + borders
        new("depthwise-wide-s2", 6, 13, 70, 6, 3, 3, [2, 2], [1, 1, 1, 1], [1, 1], 6, true),   // even-lane permute
        new("depthwise-wide-5x5-s2", 5, 12, 77, 5, 5, 5, [2, 2], [2, 2, 2, 2], [1, 1], 5, false),
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
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(c.C, c.H, c.W));
        var wv = SpatialKit.Weight(tg, "w", w, [c.O, Cg, c.KH, c.KW], wtype, out var weff);
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
        var p = Kit.P(N);
        m.Run(p);
        var exp = SRef.Conv(x, N, c.C, c.H, c.W, weff, c.O, c.KH, c.KW, bias, c.Stride, c.Pad, c.Dil, c.Groups, out _, out _);
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4, $"{name} N={N} {wtype}");
    }

    [Fact]
    public void ConvOnStridedAndChainedInputs()
    {
        // NHWC -> NCHW permute feeding a conv, then conv -> relu -> conv (activation weights stay in the arena)
        int N = 2;
        int H = 9;
        int W = 10;
        int C = 5;
        int O = 6;
        var x = Kit.Rand(N * H * W * C, 21);
        var w1 = Kit.Rand(O * C * 9, 22, 0.4f);
        var w2 = Kit.Rand(O * O, 23, 0.4f);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, H, W, C);
        var nchw = tg.B.Permute(xin, 0, 3, 1, 2);
        var w1v = tg.WF("w1", w1, O, C, 3, 3);
        var w2v = tg.WF("w2", w2, O, O, 1, 1);
        var a1 = Attributes.From(("pads", new long[] { 1, 1, 1, 1 }));
        var y1 = tg.B.Emit1("Conv", [nchw, w1v], a1);
        var y2 = tg.B.Emit1("Conv", [tg.B.Unary("Relu", y1), w2v]);
        tg.B.Output("y", y2);
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);
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

    /// <summary>
    /// Conv epilogues: residual Add + Clip, the Sigmoid·x (SiLU) pattern, and a Relu whose output is also a graph output,
    /// on the tile-packed, im2col and depthwise paths, against the unfused reference.
    /// </summary>
    [Theory]
    [InlineData(5, 3, 1)]     // tile-packed 3x3
    [InlineData(160, 3, 1)]   // K = 1440: im2col panels
    [InlineData(6, 3, 6)]     // depthwise
    [InlineData(16, 1, 1)]    // 1x1 direct
    public void ConvEpilogues(int C, int k, int groups)
    {
        int N = 2;
        int H = 9;
        int W = 37;
        int O = groups == 1 ? 8 : C;
        int pad = k / 2;
        var x = Kit.Rand(N * C * H * W, 31);
        var w = Kit.Rand(O * (C / groups) * k * k, 32, 0.3f);
        var r = Kit.Rand(N * O * H * W, 33);
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(C, H, W));
        var rin = tg.InF("r", SpatialKit.DSym(O, H, W));
        var wv = tg.WF("w", w, O, C / groups, k, k);
        var attrs = Attributes.From(("groups", (long)groups), ("pads", new long[] { pad, pad, pad, pad }));
        var clipped = tg.B.Emit1("Clip", [tg.B.Add(tg.B.Emit1("Conv", [xin, wv], attrs), rin)], Attributes.From(("min", 0f), ("max", 6f)));
        var c2 = tg.B.Emit1("Conv", [xin, wv], attrs);
        var silu = tg.B.Mul(c2, tg.B.Unary("Sigmoid", c2));
        var relu = tg.B.Unary("Relu", tg.B.Emit1("Conv", [xin, wv], attrs));
        tg.B.Output("clip", clipped);
        tg.B.Output("silu", silu);
        tg.B.Output("relu", relu);
        tg.B.Output("relu2", tg.B.Add(relu, relu));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2);
        Kit.Set(m, xin, x);
        Kit.Set(m, rin, r);
        var p = Kit.P(N);
        m.Run(p);
        var conv = SRef.Conv(x, N, C, H, W, w, O, k, k, null, [1, 1], [pad, pad, pad, pad], [1, 1], groups, out _, out _);
        Kit.AssertClose(conv.Select((v, i) => Math.Clamp(v + r[i], 0f, 6f)).ToArray(), Kit.Out(m, "clip", p), 1e-4, 1e-4, "clip");
        Kit.AssertClose(conv.Select(v => v / (1f + MathF.Exp(-v))).ToArray(), Kit.Out(m, "silu", p), 1e-4, 1e-4, "silu");
        Kit.AssertClose(conv.Select(v => Math.Max(0f, v)).ToArray(), Kit.Out(m, "relu", p), 1e-4, 1e-4, "relu");
        Kit.AssertClose(conv.Select(v => 2 * Math.Max(0f, v)).ToArray(), Kit.Out(m, "relu2", p), 1e-4, 1e-4, "relu2");
    }

    [Fact]
    public void SupportsReasons()
    {
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(4, 8, 8));
        var xin2 = tg.InF("w", SpatialKit.D(4, 4, 3, 3)); // activation as weight
        tg.B.Output("y", tg.B.Emit1("Conv", [xin, xin2]));
        var g = tg.B.Build();
        var node = g.Nodes.First(n => n.Op.Name == "Conv");
        var s = Kit.Device.Supports(node);
        Assert.False(s.IsSupported);
        Assert.Contains("weight", s.Reason);
    }
}
