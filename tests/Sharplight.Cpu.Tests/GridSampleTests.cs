using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

public class GridSampleTests
{
    [Theory, InlineData("linear"), InlineData("nearest"), InlineData("cubic")]
    public void IdentityGridWithAlignCornersReproducesInput(string mode)
    {
        int N = 2;
        int C = 3;
        int H = 5;
        int W = 6;
        var x = Kit.Rand(N * C * H * W, 141);
        var grid = new float[N * H * W * 2];
        for (int n = 0; n < N; n++)
        {
            for (int i = 0; i < H; i++)
            {
                for (int j = 0; j < W; j++)
                {
                    int o = ((n * H + i) * W + j) * 2;
                    grid[o] = -1f + 2f * j / (W - 1);
                    grid[o + 1] = -1f + 2f * i / (H - 1);
                }
            }
        }

        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(C, H, W));
        var gin = tg.InF("g", SpatialKit.DSym(H, W, 2));
        tg.B.Output("y", tg.B.Emit1("GridSample", [xin, gin], GeneralKit.A(("mode", mode), ("align_corners", true), ("padding", "border"))));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 2);
        Kit.Set(m, xin, x);
        Kit.Set(m, gin, grid);
        var p = Kit.P(N);
        m.Run(p);
        Kit.AssertClose(x, Kit.Out(m, "y", p), 1e-5, 1e-5, mode);
    }

    [Fact]
    public void ZerosPaddingOutsideAndKnownBilinear()
    {
        var x = new float[] { 1, 2, 3, 4 };  // [1,1,2,2]
        var grid = new float[] { 0f, 0f, -3f, 0f, -1f, -1f };  // center (align_corners=false), far outside, top-left corner
        var tg = new TestGraph();
        var xin = tg.InF("x", 1, 1, 2, 2);
        var gin = tg.InF("g", 1, 1, 3, 2);
        tg.B.Output("y", tg.B.Emit1("GridSample", [xin, gin], GeneralKit.A(("mode", "linear"), ("padding", "zeros"), ("align_corners", false))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        Kit.Set(m, gin, grid);
        var p = Kit.P(1);
        m.Run(p);
        // center = mean of all four; far outside = 0; (-1,-1) with align_corners=false is at pixel coord (-0.5,-0.5): 1/4 weight on pixel (0,0)
        Kit.AssertClose([2.5f, 0f, 0.25f], Kit.Out(m, "y", p), 1e-6, 1e-6);
    }

    [Fact]
    public void ReportsSupportForGeneralOps()
    {
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(4, 8));
        var ix = tg.In("i", DataType.I32, 3);
        tg.B.Output("a", tg.B.Emit1("GatherAxis", [xin, ix], GeneralKit.A(("axis", 1L))));
        tg.B.Output("b", tg.B.Emit1("CumSum", [xin], GeneralKit.A(("axis", 1L))));
        tg.B.Output("c", tg.B.Emit1("Where", [xin, xin, xin]));
        tg.B.Output("d", tg.B.Emit1("Mod", [xin, xin]));
        foreach (var n in tg.B.Build().Nodes.Where(n => n.Op.Category != OpCategory.View))
        {
            Assert.True(Kit.Device.Supports(n).IsSupported, n.Op.Name);
        }

        tg = new TestGraph();
        xin = tg.InF("x", SpatialKit.DSym(4, 8));
        var bad = tg.InF("i", 3);
        tg.B.Output("a", tg.B.Emit1("GatherAxis", [xin, bad], GeneralKit.A(("axis", 1L))));
        var s = Kit.Device.Supports(tg.B.Build().Nodes.Last());
        Assert.False(s.IsSupported);
        Assert.Contains("indices", s.Reason);
    }
}
