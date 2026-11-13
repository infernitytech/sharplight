using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

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
        var tg = new TestGraph();
        Value xin;
        Value src;
        float[] logical = x;
        if (permuted)
        {
            long N = dims[0];
            long C = dims[1];
            long H = dims[2];
            long W = dims[3];
            xin = tg.InF("x", SpatialKit.D(N, H, W, C));
            src = tg.B.Permute(xin, 0, 3, 1, 2);
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
        else
        {
            xin = tg.InF("x", SpatialKit.D(dims));
            src = xin;
        }
        tg.B.Output("y", tg.B.Emit1("Pad", [src], Attributes.From(("pads", pads), ("mode", mode), ("value", 1.5f))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(1);
        m.Run(p);
        var exp = SRef.Pad(logical, dims, pads, mode, 1.5f, out var od);
        Assert.Equal(od, m.Graph.Outputs["y"].Shape.Dims.Select(d => d.Fixed).ToArray());
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 0, 0, $"{mode} pads {string.Join(",", pads)}");
    }

    [Fact]
    public void PadSymbolicBatch()
    {
        int N = 3;
        var x = Kit.Rand(N * 2 * 4 * 5, 62);
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(2, 4, 5));
        tg.B.Output("y", tg.B.Emit1("Pad", [xin], Attributes.From(("pads", new long[] { 0, 0, 1, 2, 0, 0, 1, 1 }), ("mode", "edge"))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);
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
        int N = 2;
        int C = 3;
        int H = 4;
        int W = 5;
        var x = Kit.Rand(N * C * H * W, 63);
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(C, H, W));
        tg.B.Output("y", tg.B.Emit1("Resize", [xin], Attributes.From(("size", new long[] { oh, ow }), ("mode", mode), ("coord", coord), ("nearest_mode", nearest))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);
        var exp = SRef.Resize(x, N, C, H, W, oh, ow, mode, coord, nearest);
        Kit.AssertClose(exp, Kit.Out(m, "y", p), mode == "nearest" ? 0 : 1e-6, 1e-6, $"{mode} {coord} {nearest} {oh}x{ow}");
    }
}
