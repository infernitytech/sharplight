using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

public class SupportTests
{
    private static Node NodeOf(TestGraph tg) => tg.B.Build().Nodes.Last();

    [Fact]
    public void ReportsReasonsForUnsupportedAttributeCombos()
    {
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(4, 8, 8));
        tg.B.Output("y", tg.B.Emit1("Pool", [xin], Attributes.From(("mode", "avg"), ("kernel", new long[] { 2, 2 }), ("dilations", new long[] { 2, 2 }))));
        var s = Kit.Device.Supports(NodeOf(tg));
        Assert.True(s.IsSupported);   // dilated average pooling is supported since the include-pad count follows the taps

        tg = new TestGraph();
        xin = tg.InF("x", SpatialKit.DSym(4, 8, 8));
        tg.B.Output("y", tg.B.Emit1("Resize", [xin], Attributes.From(("size", new long[] { 4, 4 }), ("mode", "cubic"))));
        s = Kit.Device.Supports(NodeOf(tg));
        Assert.False(s.IsSupported);
        Assert.Contains("cubic", s.Reason);

        tg = new TestGraph();
        xin = tg.InF("x", SpatialKit.DSym(4, 8, 8));
        tg.B.Output("y", tg.B.Emit1("Pad", [xin], Attributes.From(("pads", new long[] { 0, 0, 0, 0, 0, 0, 0, 0 }), ("mode", "wrap"))));
        s = Kit.Device.Supports(NodeOf(tg));
        Assert.False(s.IsSupported);
        Assert.Contains("wrap", s.Reason);
    }

    [Fact]
    public void ReportsSupportForNewOps()
    {
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(4, 8, 8));
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
