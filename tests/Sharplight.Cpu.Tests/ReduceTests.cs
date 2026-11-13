using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

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
        var tg = new TestGraph();
        long total = SRef.Prod(dims);
        var x = Kit.Rand((int)total, 41);
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
        else
        {
            xin = tg.InF("x", SpatialKit.D(dims));
            src = xin;
        }
        tg.B.Output("y", tg.B.Emit1("Reduce", [src], Attributes.From(("mode", mode), ("axes", axes), ("keepdims", keep))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(1);
        m.Run(p);
        var exp = SRef.Reduce(logical, dims, axes, mode, keep, out var od);
        var shape = m.Graph.Outputs["y"].Shape;
        Assert.Equal(od, shape.Dims.Select(d => d.Fixed).ToArray());
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-5, $"{mode} {string.Join(",", dims)} axes {string.Join(",", axes)}");
    }

    [Fact]
    public void ReduceWithSymbolicBatch()
    {
        int N = 3;
        int C = 24;
        int H = 7;
        int W = 7;
        var x = Kit.Rand(N * C * H * W, 42);
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(C, H, W));
        tg.B.Output("gap", tg.B.Emit1("Reduce", [xin], Attributes.From(("mode", "mean"), ("axes", new long[] { 2, 3 }), ("keepdims", false))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);
        Kit.AssertClose(SRef.Reduce(x, [N, C, H, W], [2, 3], "mean", false, out _), Kit.Out(m, "gap", p), 1e-5, 1e-5);
    }
}
