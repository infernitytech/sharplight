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

    /// <summary>
    /// Rank-3 tables gather whole [H, W] rows (#111: only the first W values were written, the rest kept stale arena
    /// data, so the second run differed). Two runs with different table contents; W = 8 takes the 8-value-group kernel,
    /// W = 6 the one-thread-per-value one, a sliced table the dense copy first.
    /// </summary>
    [Theory]
    [InlineData(8, false)]
    [InlineData(6, false)]
    [InlineData(8, true)]
    public void Rank3ActivationTable(int w, bool sliced)
    {
        Gpu.Require();
        var tg = new TG();
        int h = 3;
        int wIn = sliced ? w + 8 : w;
        var x = tg.InF("x", 5, h, wIn);
        var rows = tg.In("rows", DataType.I32, Dim.Sym("R"));
        var table = sliced ? tg.B.Slice(x, 2, 4, w) : x;
        tg.B.Output("y", tg.B.Gather(table, rows));
        var g = tg.B.Build();
        Diff.Check($"Gather.rank3.{w}{(sliced ? ".view" : "")}", g,
        [
            new Step(1, 0, (x, Rnd.F(5 * h * wIn, 1)), (rows, new[] { 4, 0, 2 })) { R = 3 },
            new Step(1, 0, (x, Rnd.F(5 * h * wIn, 2)), (rows, new[] { 1, 3, 3, 0 })) { R = 4 },
        ], atol: 1e-6);
    }

    [Theory]
    [InlineData("f16")]
    [InlineData("q8_0")]
    public void Rank3WeightTable(string fmt)
    {
        Gpu.Require();
        var tg = new TG();
        var ids = tg.In("ids", DataType.I32, TG.T);
        var table = tg.RandW("emb", DataType.Get(fmt), 1f, 20, 2, 64);
        tg.B.Output("y", tg.B.Gather(table, ids));
        Diff.Check("Gather.rank3." + fmt, tg.B.Build(),
            [new Step(5, 0, (ids, Rnd.I(5, 4, 20))), new Step(7, 0, (ids, Rnd.I(7, 5, 20)))], atol: 1e-6, rtol: 1e-6);
    }
}
