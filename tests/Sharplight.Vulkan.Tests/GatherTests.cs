using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class GatherTests
{
    [Theory]
    [InlineData("f32", 128)]
    [InlineData("f16", 128)]
    [InlineData("bf16", 64)]
    [InlineData("q8_0", 128)]
    [InlineData("q4_0", 96)]
    [InlineData("q4_k", 256)]
    [InlineData("q5_k", 512)]
    [InlineData("q6_k", 256)]
    public void Table(string fmt, int d)
    {
        Gpu.Require();
        int V = 50;
        var tg = new TG();
        var ids = tg.In("ids", DataType.I32, TG.T);
        var table = tg.RandW("emb", DataType.Get(fmt), 1f, V, d);
        tg.B.Output("y", tg.B.Gather(table, ids));
        Diff.Check("Gather." + fmt, tg.B.Build(), new Step(9, 0, (ids, Rnd.I(9, 4, V))), atol: 1e-6, rtol: 1e-6);
    }

    [Fact]
    public void ActivationTableWithSlicedRows()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 32);
        var rows = tg.In("rows", DataType.I32, Dim.Sym("R"));
        var s = tg.B.Slice(x, 1, 8, 16);                       // rows have stride 32, 16 wide
        tg.B.Output("y", tg.B.Gather(s, rows));
        var g = tg.B.Build();
        Diff.Check("Gather.view", g, new Step(6, 0, (x, Rnd.F(6 * 32, 1)), (rows, new[] { 5, 1, 3 })) { R = 3 }, atol: 1e-6);
    }
}
