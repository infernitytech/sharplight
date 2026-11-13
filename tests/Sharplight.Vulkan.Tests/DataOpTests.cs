using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class DataOpTests
{
    [Fact]
    public void ExpandAndTile()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 1, 4);
        tg.B.Output("e", tg.B.Emit1("Expand", [x], Attributes.From(("shape", new long[] { 1, 5, 4 }))));
        tg.B.Output("t", tg.B.Emit1("Tile", [x], Attributes.From(("repeats", new long[] { 1, 3, 2 }))));
        var r = tg.InF("r", 3, 1);
        tg.B.Output("e2", tg.B.Emit1("Expand", [r], Attributes.From(("shape", new long[] { 2, 3, 6 }))));
        Diff.Check("Expand+Tile", tg.B.Build(), new Step(3, 0, (x, Rnd.F(12, 1)), (r, Rnd.F(3, 2))), atol: 0, rtol: 0);
    }

    [Theory]
    [InlineData(0, 3)]
    [InlineData(1, 3)]
    [InlineData(-1, 2)]
    [InlineData(1, 0)]
    public void GatherAxis(int axis, int idxRank)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", 6, 5, 4);
        int rank = idxRank == 0 ? 0 : idxRank == 3 ? 1 : 2;
        var shape = rank == 0 ? [] : rank == 1 ? new Dim[] { 3 } : [2, 3];
        var idx = tg.In("idx", DataType.I32, shape);
        tg.B.Output("y", tg.B.Emit1("GatherAxis", [x, idx], Attributes.From(("axis", (long)axis))));
        int a = axis < 0 ? axis + 3 : axis;
        int n = rank == 0 ? 1 : rank == 1 ? 3 : 6;
        var vals = Rnd.I(n, 3, 4).Select(v => v - (v % 2 == 0 ? 4 : 0)).ToArray();   // some negative indices
        Diff.Check("GatherAxis", tg.B.Build(), new Step(1, 0, (x, Rnd.F(120, 1)), (idx, vals)), atol: 0, rtol: 0, what: $"axis {axis} rank {rank}");
    }

    [Fact]
    public void GatherAxisInt64IndicesSymbolicData()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 7);
        var idx = tg.In("idx", DataType.I64, 4);
        tg.B.Output("y", tg.B.Emit1("GatherAxis", [x, idx], Attributes.From(("axis", 1L))));
        Diff.Check("GatherAxis.i64", tg.B.Build(), new Step(5, 0, (x, Rnd.F(35, 1)), (idx, new long[] { 6, 0, -1, 3 })), atol: 0, rtol: 0);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(-1)]
    public void GatherElements(int axis)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", 4, 5, 6);
        var idx = tg.In("idx", DataType.I32, 3, 5, 6);
        int a = axis < 0 ? axis + 3 : axis;
        long lim = new long[] { 4, 5, 6 }[a];
        tg.B.Output("y", tg.B.Emit1("GatherElements", [x, idx], Attributes.From(("axis", (long)axis))));
        Diff.Check("GatherElements", tg.B.Build(), new Step(1, 0, (x, Rnd.F(120, 1)), (idx, Rnd.I(90, 2, (int)lim))), atol: 0, rtol: 0, what: $"axis {axis}");
    }

    [Theory]
    [InlineData(0, 2)]
    [InlineData(0, 1)]
    [InlineData(1, 1)]
    [InlineData(0, 3)]
    public void GatherND(int batchDims, int q)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", 3, 4, 5);
        int[] shapeI = batchDims == 0 ? [6, q] : [3, 2, q];
        var idx = tg.In("idx", DataType.I32, [.. shapeI.Select(v => (Dim)v)]);
        tg.B.Output("y", tg.B.Emit1("GatherND", [x, idx], Attributes.From(("batch_dims", (long)batchDims))));
        long[] dims = [3, 4, 5];
        var vals = new int[shapeI.Aggregate(1, (a, b) => a * b)];
        var rng = new Random(4);
        for (int i = 0; i < vals.Length; i++)
        {
            int j = i % q + batchDims;
            vals[i] = rng.Next((int)dims[j]);
        }
        Diff.Check("GatherND", tg.B.Build(), new Step(1, 0, (x, Rnd.F(60, 1)), (idx, vals)), atol: 0, rtol: 0, what: $"b={batchDims} q={q}");
    }

    [Theory]
    [InlineData("none", 1)]
    [InlineData("add", 1)]
    [InlineData("mul", 0)]
    [InlineData("max", 1)]
    [InlineData("min", 0)]
    [InlineData("add", 0)]
    public void ScatterElements(string red, int axis)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", 4, 6);
        var idx = tg.In("idx", DataType.I32, 3, 4);
        var u = tg.InF("u", 3, 4);
        long lim = axis == 1 ? 6 : 4;
        tg.B.Output("y", tg.B.Emit1("ScatterElements", [x, idx, u], Attributes.From(("axis", (long)axis), ("reduction", red))));
        // unique indices for "none" along the axis so the result is order independent
        int[] iv = axis == 1 ? [0, 2, 5, 1, 3, 4, 5, 0, 2, 1, 4, 3] : [1, 3, 0, 2, 3, 1, 0, 2, 0, 1, 2, 3];
        if (red != "none")
        {
            iv = Rnd.I(12, 7, (int)lim);
        }

        Diff.Check("ScatterElements", tg.B.Build(), new Step(1, 0, (x, Rnd.F(24, 1)), (idx, iv), (u, Rnd.F(12, 2))), atol: 1e-5, what: $"{red} axis {axis}");
    }

    [Theory]
    [InlineData("none")]
    [InlineData("add")]
    [InlineData("max")]
    public void ScatterND(string red)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", 4, 5, 3);
        var idx = tg.In("idx", DataType.I32, 4, 2);
        var u = tg.InF("u", 4, 3);
        tg.B.Output("y", tg.B.Emit1("ScatterND", [x, idx, u], Attributes.From(("reduction", red))));
        var iv = new[] { 0, 1, 1, 2, 2, 4, 3, 0 };   // unique rows
        if (red != "none")
        {
            iv = [0, 1, 0, 1, 2, 2, 0, 1];
        }

        Diff.Check("ScatterND", tg.B.Build(), new Step(1, 0, (x, Rnd.F(60, 1)), (idx, iv), (u, Rnd.F(12, 2))), atol: 1e-5, what: red);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void CumSum(bool exclusive, bool reverse)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 5, 9);
        tg.B.Output("a1", tg.B.Emit1("CumSum", [x], Attributes.From(("axis", 2L), ("exclusive", exclusive), ("reverse", reverse))));
        tg.B.Output("a0", tg.B.Emit1("CumSum", [x], Attributes.From(("axis", 1L), ("exclusive", exclusive), ("reverse", reverse))));
        Diff.Check("CumSum", tg.B.Build(), new Step(3, 0, (x, Rnd.F(135, 1))), atol: 1e-5, what: $"excl={exclusive} rev={reverse}");
    }

    [Theory]
    [InlineData("max", 1, true, false)]
    [InlineData("min", 1, true, false)]
    [InlineData("max", 2, false, false)]
    [InlineData("max", 1, true, true)]
    [InlineData("min", 2, false, true)]
    public void ArgReduce(string mode, int axis, bool keep, bool last)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 7, 6);
        tg.B.Output("y", tg.B.Emit1("ArgReduce", [x], Attributes.From(("mode", mode), ("axis", (long)axis), ("keepdims", keep), ("select_last_index", last))));
        var data = Rnd.F(3 * 42, 3).Select(v => MathF.Round(v * 3)).ToArray();    // many ties
        Diff.Check("ArgReduce", tg.B.Build(), new Step(3, 0, (x, data)), what: $"{mode} axis {axis} keep={keep} last={last}");
    }

    [Theory]
    [InlineData(true, 1, 3)]
    [InlineData(false, 1, 4)]
    [InlineData(true, 2, 5)]
    [InlineData(true, 1, 7)]
    public void TopK(bool largest, int axis, int k)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 7, 9);
        var outs = tg.B.Emit("TopK", [x], Attributes.From(("k", (long)k), ("axis", (long)axis), ("largest", largest)));
        tg.B.Output("v", outs[0]); tg.B.Output("i", outs[1]);
        var data = Rnd.F(2 * 63, 3).Select(v => MathF.Round(v * 4) / 4).ToArray();   // ties
        Diff.Check("TopK", tg.B.Build(), new Step(2, 0, (x, data)), what: $"largest={largest} axis={axis} k={k}");
    }

    [Theory]
    [InlineData(0, 1, 4)]
    [InlineData(0, -1, 4)]
    [InlineData(1, 2, 5)]
    public void Trilu(int k, int unused, int n)
    {
        Gpu.Require();
        foreach (var upper in new[] { true, false })
        {
            var tg = new TG();
            var x = tg.InF("x", TG.T, n, n + 2);
            tg.B.Output("y", tg.B.Emit1("Trilu", [x], Attributes.From(("upper", upper), ("k", (long)(k + unused)))));
            Diff.Check("Trilu", tg.B.Build(), new Step(2, 0, (x, Rnd.F(2 * n * (n + 2), 1))), atol: 0, what: $"upper={upper} k={k + unused}");
        }
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(1)]
    public void OneHot(int axis)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.In("x", DataType.I32, TG.T, 4);
        tg.B.Output("y", tg.B.Emit1("OneHot", [x], Attributes.From(("depth", 6L), ("axis", (long)axis), ("on", 2f), ("off", -1f))));
        Diff.Check("OneHot", tg.B.Build(), new Step(3, 0, (x, new[] { 0, 5, -1, 3, 2, 2, 4, 1, 0, 0, 5, 3 })), atol: 0, what: $"axis {axis}");
    }

    /// <summary>
    /// Rows of a weight table selected by indices (negative ones wrap) and consumed only by one elementwise group: the group reads
    /// the rows in place (no gather launch). The second output keeps a materialized gather next to the folded one.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RowGatherReadInElementwiseGroup(bool alsoOutput)
    {
        Gpu.Require();
        var tg = new TG();
        var table = tg.RandW("table", DataType.F32, 1f, 5, 24);
        var x = tg.InF("x", TG.T, 2, 24);
        var idx = tg.In("idx", DataType.I32, TG.T, 2);
        var rows = tg.B.Emit1("GatherAxis", [table, idx], Attributes.From(("axis", 0L)));
        tg.B.Output("y", tg.B.Mul(tg.B.Add(rows, x), x));
        if (alsoOutput)
        {
            tg.B.Output("rows", rows);
        }

        int t = 7;
        var ids = Enumerable.Range(0, t * 2).Select(i => i % 7 - 2).ToArray();   // -2 .. 4
        Diff.Check("GatherAxis.fold", tg.B.Build(), new Step(t, 0, (x, Rnd.F(t * 2 * 24, 5)), (idx, ids)), atol: 1e-6, rtol: 1e-6);
    }
}
