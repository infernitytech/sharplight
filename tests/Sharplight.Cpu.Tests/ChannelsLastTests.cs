using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// The channels-last conv, depthwise, pooling and elementwise paths (<see cref="Runtime.DeviceFeatures.ChannelsLast"/>):
/// inputs arrive as NHWC tensors permuted to NCHW (a channels-last view), so every op of the chain runs channels-last;
/// results are compared with the f64-accumulated reference on the NCHW data.
/// </summary>
public class ChannelsLastTests
{
    public record Case(string Name, int C, int H, int W, int O, int KH, int KW, int[] Stride, int[] Pad, int[] Dil, int Groups, bool Bias);

    private static readonly Case[] Cases =
    [
        new("3x3-c5-rows", 5, 11, 13, 7, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, true),
        new("7x7-s2-c3-rows", 3, 33, 31, 16, 7, 7, [2, 2], [3, 3, 3, 3], [1, 1], 1, true),
        new("3x3-c20-taps", 20, 11, 13, 40, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, true),
        new("3x3-c33-o33", 33, 9, 10, 33, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, false),
        new("3x3-s2-c24", 24, 14, 15, 70, 3, 3, [2, 2], [1, 1, 1, 1], [1, 1], 1, true),
        new("1x1-c64-o20", 64, 7, 9, 20, 1, 1, [1, 1], [0, 0, 0, 0], [1, 1], 1, true),
        new("1x1-s2-c32", 32, 9, 9, 64, 1, 1, [2, 2], [0, 0, 0, 0], [1, 1], 1, true),
        new("asym-pads-c16", 16, 10, 12, 6, 3, 3, [1, 1], [1, 2, 0, 3], [1, 1], 1, true),
        new("dil2-c18", 18, 15, 14, 6, 3, 3, [1, 1], [2, 2, 2, 2], [2, 2], 1, true),
        new("dil2-c4-taps", 4, 15, 14, 6, 3, 3, [1, 1], [2, 2, 2, 2], [2, 3], 1, true),
        new("rect-kernel-c17", 17, 12, 15, 5, 3, 5, [1, 2], [1, 2, 1, 2], [1, 1], 1, true),
        new("long-K", 160, 9, 9, 8, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, true),
        new("3x3-c32-o16", 32, 11, 13, 16, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, true),
        new("3x3-s2-c3-o12-rows", 3, 21, 19, 12, 3, 3, [2, 2], [1, 1, 1, 1], [1, 1], 1, true),
        new("chunked-K-s3-o36", 256, 20, 20, 36, 3, 3, [3, 3], [1, 1, 1, 1], [1, 1], 1, true),
        new("chunked-K-wide-c300", 300, 6, 7, 70, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, false),
        new("dw-3x3-c40", 40, 14, 15, 40, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 40, true),
        new("dw-3x3-s2-c24", 24, 14, 15, 24, 3, 3, [2, 2], [1, 1, 1, 1], [1, 1], 24, false),
        new("dw-5x5-dil2-c9", 9, 17, 18, 9, 5, 5, [1, 1], [4, 4, 4, 4], [2, 2], 9, true),
        new("dw-5x5-s2-c35", 35, 12, 37, 35, 5, 5, [2, 2], [2, 2, 2, 2], [1, 1], 35, true),
        new("dw-3x3-wide-c16", 16, 5, 41, 16, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 16, true),
    ];

    public static IEnumerable<object[]> Params()
    {
        foreach (var c in Cases)
        {
            foreach (var n in new[] { 1, 3 })
            {
                yield return [c.Name, n];
            }
        }
    }

    [Theory, MemberData(nameof(Params))]
    public void Conv(string name, int N)
    {
        var c = Cases.Single(k => k.Name == name);
        int cg = c.C / c.Groups;
        var w = Kit.Rand(c.O * cg * c.KH * c.KW, 11, 0.5f);
        var x = Kit.Rand(N * c.H * c.W * c.C, 12);
        var bias = c.Bias ? Kit.Rand(c.O, 13) : null;
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, c.H, c.W, c.C);
        var ins = new List<Value> { tg.B.Permute(xin, 0, 3, 1, 2), tg.WF("w", w, c.O, cg, c.KH, c.KW) };
        if (bias != null)
        {
            ins.Add(tg.WF("b", bias, c.O));
        }

        tg.B.Output("y", tg.B.Emit1("Conv", ins, ConvAttributes(c.Groups, c.Stride, c.Pad, c.Dil)));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);
        var exp = SRef.Conv(Nchw(x, N, c.C, c.H, c.W), N, c.C, c.H, c.W, w, c.O, c.KH, c.KW, bias, c.Stride, c.Pad, c.Dil, c.Groups, out _, out _);
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4, $"{name} N={N}");
    }

    /// <summary>
    /// A stem conv on NCHW input (kernel-row runs, or a layout copy for 20 channels) feeding channels-last convs with a
    /// residual Add + Clip, a SiLU and a LeakyRelu epilogue, a non-absorbed Add of two channels-last values, max and average
    /// pooling and a global mean.
    /// </summary>
    [Theory]
    [InlineData(3, 1)]
    [InlineData(3, 2)]
    [InlineData(20, 2)]
    public void Chain(int c0, int N)
    {
        const int H = 19;
        const int W = 23;
        const int C = 24;
        var x = Kit.Rand(N * c0 * H * W, 21);
        var w1 = Kit.Rand(C * c0 * 9, 22, 0.4f);
        var w2 = Kit.Rand(C * C * 9, 23, 0.2f);
        var w3 = Kit.Rand(C * 9, 24, 0.4f);
        var w4 = Kit.Rand(C * C, 25, 0.3f);
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(c0, H, W));
        var p1 = Attributes.From(("pads", new long[] { 1, 1, 1, 1 }));
        var y1 = tg.B.Unary("Relu", tg.B.Emit1("Conv", [xin, tg.WF("w1", w1, C, c0, 3, 3)], p1));
        var y2 = tg.B.Emit1("Clip", [tg.B.Add(tg.B.Emit1("Conv", [y1, tg.WF("w2", w2, C, C, 3, 3)], p1), y1)], Attributes.From(("min", 0f), ("max", 6f)));
        var c3 = tg.B.Emit1("Conv", [y2, tg.WF("w3", w3, C, 1, 3, 3)], Attributes.From(("groups", (long)C), ("pads", new long[] { 1, 1, 1, 1 })));
        var y3 = tg.B.Mul(c3, tg.B.Unary("Sigmoid", c3));
        var y4 = tg.B.Emit1("LeakyRelu", [tg.B.Emit1("Conv", [y3, tg.WF("w4", w4, C, C, 1, 1)])], Attributes.From(("alpha", 0.1f)));
        var sum = tg.B.Add(y4, y3);
        var poolMax = tg.B.Emit1("Pool", [sum], PoolAttributes("max", false));
        var poolAvg = tg.B.Emit1("Pool", [sum], PoolAttributes("avg", true));
        var mean = tg.B.Emit1("Reduce", [poolMax], Attributes.From(("mode", "mean"), ("axes", new long[] { 2, 3 }), ("keepdims", true)));
        tg.B.Output("avg", tg.B.Unary("Relu", poolAvg));
        tg.B.Output("mean", mean);
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);

        int[] one = [1, 1];
        int[] pad1 = [1, 1, 1, 1];
        var r1 = SRef.Conv(x, N, c0, H, W, w1, C, 3, 3, null, one, pad1, one, 1, out _, out _).Select(v => Math.Max(0, v)).ToArray();
        var r2 = SRef.Conv(r1, N, C, H, W, w2, C, 3, 3, null, one, pad1, one, 1, out _, out _).Select((v, i) => Math.Clamp(v + r1[i], 0f, 6f)).ToArray();
        var r3 = SRef.Conv(r2, N, C, H, W, w3, C, 3, 3, null, one, pad1, one, C, out _, out _).Select(v => v / (1f + MathF.Exp(-v))).ToArray();
        var r4 = SRef.Conv(r3, N, C, H, W, w4, C, 1, 1, null, one, [0, 0, 0, 0], one, 1, out _, out _).Select(v => v > 0 ? v : 0.1f * v).ToArray();
        var rs = r4.Select((v, i) => v + r3[i]).ToArray();
        var rMax = SRef.Pool(rs, N, C, H, W, true, [3, 3], [2, 2], pad1, one, false, false, out int ho, out int wo);
        var rAvg = SRef.Pool(rs, N, C, H, W, false, [3, 3], [2, 2], pad1, one, false, true, out _, out _).Select(v => Math.Max(0, v)).ToArray();
        var rMean = SRef.Reduce(rMax, [N, C, ho, wo], [2, 3], "mean", true, out _);
        Kit.AssertClose(rAvg, Kit.Out(m, "avg", p), 1e-4, 1e-4, "avg");
        Kit.AssertClose(rMean, Kit.Out(m, "mean", p), 1e-4, 1e-4, "mean");
    }

    /// <summary>
    /// A YOLO-style block: a channel split of a channels-last conv output (views), a conv and a SiLU written out on one part,
    /// a per-channel scale, a channel concat with the other part, and a conv over the concat.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void SplitConcat(int N)
    {
        const int H = 13;
        const int W = 17;
        const int C = 32;
        var x = Kit.Rand(N * H * W * C, 41);
        var w1 = Kit.Rand(C * C, 42, 0.3f);
        var w2 = Kit.Rand(16 * 16 * 9, 43, 0.3f);
        var w3 = Kit.Rand(24 * 48, 44, 0.3f);
        var scale = Kit.Rand(16, 45);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, H, W, C);
        var y1 = tg.B.Emit1("Conv", [tg.B.Permute(xin, 0, 3, 1, 2), tg.WF("w1", w1, C, C, 1, 1)]);
        var a = tg.B.Slice(y1, 1, 0, 16);
        var b = tg.B.Slice(y1, 1, 16, 16);
        var c2 = tg.B.Emit1("Conv", [b, tg.WF("w2", w2, 16, 16, 3, 3)], Attributes.From(("pads", new long[] { 1, 1, 1, 1 })));
        var silu = tg.B.Mul(tg.B.Unary("Sigmoid", c2), c2);
        var scaled = tg.B.Mul(silu, tg.WF("s", scale, 16, 1, 1));
        var cat = tg.B.Concat(1, a, b, scaled);
        tg.B.Output("y", tg.B.Emit1("Conv", [cat, tg.WF("w3", w3, 24, 48, 1, 1)]));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);

        int[] one = [1, 1];
        int[] none = [0, 0, 0, 0];
        int hw = H * W;
        var r1 = SRef.Conv(Nchw(x, N, C, H, W), N, C, H, W, w1, C, 1, 1, null, one, none, one, 1, out _, out _);
        var rb = new float[N * 16 * hw];
        for (int n = 0; n < N; n++)
        {
            Array.Copy(r1, (n * C + 16) * hw, rb, n * 16 * hw, 16 * hw);
        }

        var r2 = SRef.Conv(rb, N, 16, H, W, w2, 16, 3, 3, null, one, [1, 1, 1, 1], one, 1, out _, out _);
        var rs = r2.Select((v, i) => v / (1f + MathF.Exp(-v)) * scale[i / hw % 16]).ToArray();
        var cat48 = new float[N * 48 * hw];
        for (int n = 0; n < N; n++)
        {
            Array.Copy(r1, n * C * hw, cat48, n * 48 * hw, C * hw);
            Array.Copy(rs, n * 16 * hw, cat48, (n * 48 + C) * hw, 16 * hw);
        }

        var r3 = SRef.Conv(cat48, N, 48, H, W, w3, 24, 1, 1, null, one, none, one, 1, out _, out _);
        Kit.AssertClose(r3, Kit.Out(m, "y", p), 1e-4, 1e-4);
    }

    /// <summary>Resize of a channels-last view (an NHWC input seen as NCHW) writes a channels-last output of the same values.</summary>
    [Theory]
    [InlineData("nearest", "asymmetric", 2, 26, 34)]
    [InlineData("nearest", "half_pixel", 1, 7, 9)]
    [InlineData("linear", "half_pixel", 2, 26, 34)]
    [InlineData("linear", "align_corners", 1, 9, 5)]
    public void Resize(string mode, string coord, int N, int oh, int ow)
    {
        const int H = 13;
        const int W = 17;
        const int C = 20;
        var x = Kit.Rand(N * H * W * C, 51);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, H, W, C);
        var resized = tg.B.Emit1("Resize", [tg.B.Permute(xin, 0, 3, 1, 2)], Attributes.From(("size", new long[] { oh, ow }), ("mode", mode), ("coord", coord)));
        tg.B.Output("y", tg.B.Unary("Relu", resized));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);
        var exp = SRef.Resize(Nchw(x, N, C, H, W), N, C, H, W, oh, ow, mode, coord, "round_prefer_floor").Select(v => Math.Max(0, v)).ToArray();
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-6, 1e-6, $"{mode} {coord}");
    }

    private static Attributes ConvAttributes(int groups, int[] stride, int[] pad, int[] dil) => Attributes.From(
        ("groups", (long)groups), ("strides", stride.Select(v => (long)v).ToArray()), ("pads", pad.Select(v => (long)v).ToArray()),
        ("dilations", dil.Select(v => (long)v).ToArray()));

    private static Attributes PoolAttributes(string mode, bool includePad) => Attributes.From(
        ("mode", mode), ("kernel", new long[] { 3, 3 }), ("strides", new long[] { 2, 2 }), ("pads", new long[] { 1, 1, 1, 1 }),
        ("dilations", new long[] { 1, 1 }), ("ceil_mode", false), ("count_include_pad", includePad));

    private static float[] Nchw(float[] nhwc, int n, int c, int h, int w)
    {
        var r = new float[nhwc.Length];
        for (int i = 0; i < n; i++)
        {
            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    for (int k = 0; k < c; k++)
                    {
                        r[((i * c + k) * h + y) * w + x] = nhwc[((i * h + y) * w + x) * c + k];
                    }
                }
            }
        }

        return r;
    }
}
