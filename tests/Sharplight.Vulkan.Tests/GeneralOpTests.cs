using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public class CastAndIntegerTests
{
    [Theory]
    [InlineData("f16")] [InlineData("bf16")] [InlineData("i32")] [InlineData("i64")]
    public void CastRoundTrip(string t)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 19);
        var c = tg.B.Emit1("Cast", [x], Attributes.From(("type", t)));
        tg.B.Output("back", tg.B.Emit1("Cast", [c], Attributes.From(("type", "f32"))));
        if (t == "i32" || t == "i64") tg.B.Output("raw", c);
        var data = Rnd.F(3 * 19, 5, 100f); data[0] = -7.9f; data[1] = 7.9f; data[2] = 0.5f;
        Diff.Check("Cast." + t, tg.B.Build(), new Step(3, 0, (x, data)), atol: 0, rtol: 0);
    }

    [Fact]
    public void CastBetweenIntegerTypes()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.In("x", DataType.I64, TG.T, 5);
        var i = tg.B.Emit1("Cast", [x], Attributes.From(("type", "i32")));
        tg.B.Output("i32", i);
        tg.B.Output("f", tg.B.Emit1("Cast", [i], Attributes.From(("type", "f32"))));
        tg.B.Output("i64", tg.B.Emit1("Cast", [i], Attributes.From(("type", "i64"))));
        Diff.Check("Cast.int", tg.B.Build(), new Step(3, 0, (x, Enumerable.Range(-7, 15).Select(v => (long)v * 3).ToArray())), atol: 0, rtol: 0);
    }

    [Theory] [InlineData("i32")] [InlineData("i64")]
    public void IntegerTensorsThroughViewsConcatGatherWhereExpand(string t)
    {
        Gpu.Require();
        var dt = DataType.Get(t);
        var tg = new TG(); var x = tg.In("x", dt, TG.T, 6); var y = tg.In("y", dt, TG.T, 6);
        var s = tg.B.Slice(x, 1, 1, 4);                             // [T,4] strided
        var pm = tg.B.Permute(x, 0, 1);
        tg.B.Output("slice", s);                                     // view as graph output
        tg.B.Output("cat", tg.B.Concat(1, s, tg.B.Slice(y, 1, 0, 2)));
        tg.B.Output("contig", tg.B.Contiguous(tg.B.Slice(y, 1, 2, 3)));
        var cond = tg.B.Binary("Greater", tg.B.Emit1("Cast", [x], Attributes.From(("type", "f32"))), tg.Const("zero", [0f], 1));
        tg.B.Output("where", tg.B.Emit1("Where", [cond, x, y]));
        tg.B.Output("expand", tg.B.Emit1("Expand", [tg.B.Slice(x, 1, 0, 1)], Attributes.From(("shape", new long[] { 1, 7 }))));
        var idx = tg.In("idx", DataType.I32, 3);
        tg.B.Output("gather", tg.B.Emit1("GatherAxis", [x, idx], Attributes.From(("axis", 1L))));
        Array Ints(int n, int seed) => t == "i32" ? Rnd.I(n, seed, 50).Select(v => v - 25).ToArray() : Rnd.I(n, seed, 50).Select(v => (long)v - 25).ToArray();
        Diff.Check("Integer.flow." + t, tg.B.Build(), new Step(4, 0, (x, Ints(24, 1)), (y, Ints(24, 2)), (idx, new[] { 5, 0, 3 })), atol: 0, rtol: 0);
    }
}

public class DataOpTests
{
    [Fact]
    public void ExpandAndTile()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 1, 4);
        tg.B.Output("e", tg.B.Emit1("Expand", [x], Attributes.From(("shape", new long[] { 1, 5, 4 }))));
        tg.B.Output("t", tg.B.Emit1("Tile", [x], Attributes.From(("repeats", new long[] { 1, 3, 2 }))));
        var r = tg.InF("r", 3, 1);
        tg.B.Output("e2", tg.B.Emit1("Expand", [r], Attributes.From(("shape", new long[] { 2, 3, 6 }))));
        Diff.Check("Expand+Tile", tg.B.Build(), new Step(3, 0, (x, Rnd.F(12, 1)), (r, Rnd.F(3, 2))), atol: 0, rtol: 0);
    }

    [Theory]
    [InlineData(0, 3)] [InlineData(1, 3)] [InlineData(-1, 2)] [InlineData(1, 0)]
    public void GatherAxis(int axis, int idxRank)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", 6, 5, 4);
        int rank = idxRank == 0 ? 0 : idxRank == 3 ? 1 : 2;
        var shape = rank == 0 ? [] : rank == 1 ? new Dim[] { 3 } : [2, 3];
        var idx = tg.In("idx", DataType.I32, shape);
        tg.B.Output("y", tg.B.Emit1("GatherAxis", [x, idx], Attributes.From(("axis", (long)axis))));
        int a = axis < 0 ? axis + 3 : axis; int n = rank == 0 ? 1 : rank == 1 ? 3 : 6;
        var vals = Rnd.I(n, 3, 4).Select(v => v - (v % 2 == 0 ? 4 : 0)).ToArray();   // some negative indices
        Diff.Check("GatherAxis", tg.B.Build(), new Step(1, 0, (x, Rnd.F(120, 1)), (idx, vals)), atol: 0, rtol: 0, what: $"axis {axis} rank {rank}");
    }

    [Fact]
    public void GatherAxisInt64IndicesSymbolicData()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 7);
        var idx = tg.In("idx", DataType.I64, 4);
        tg.B.Output("y", tg.B.Emit1("GatherAxis", [x, idx], Attributes.From(("axis", 1L))));
        Diff.Check("GatherAxis.i64", tg.B.Build(), new Step(5, 0, (x, Rnd.F(35, 1)), (idx, new long[] { 6, 0, -1, 3 })), atol: 0, rtol: 0);
    }

    [Theory] [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(-1)]
    public void GatherElements(int axis)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", 4, 5, 6);
        var idx = tg.In("idx", DataType.I32, 3, 5, 6);
        int a = axis < 0 ? axis + 3 : axis; long lim = new long[] { 4, 5, 6 }[a];
        tg.B.Output("y", tg.B.Emit1("GatherElements", [x, idx], Attributes.From(("axis", (long)axis))));
        Diff.Check("GatherElements", tg.B.Build(), new Step(1, 0, (x, Rnd.F(120, 1)), (idx, Rnd.I(90, 2, (int)lim))), atol: 0, rtol: 0, what: $"axis {axis}");
    }

    [Theory] [InlineData(0, 2)] [InlineData(0, 1)] [InlineData(1, 1)] [InlineData(0, 3)]
    public void GatherND(int batchDims, int q)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", 3, 4, 5);
        int[] shapeI = batchDims == 0 ? [6, q] : [3, 2, q];
        var idx = tg.In("idx", DataType.I32, [.. shapeI.Select(v => (Dim)v)]);
        tg.B.Output("y", tg.B.Emit1("GatherND", [x, idx], Attributes.From(("batch_dims", (long)batchDims))));
        long[] dims = [3, 4, 5];
        var vals = new int[shapeI.Aggregate(1, (a, b) => a * b)];
        var rng = new Random(4);
        for (int i = 0; i < vals.Length; i++) { int j = i % q + batchDims; vals[i] = rng.Next((int)dims[j]); }
        Diff.Check("GatherND", tg.B.Build(), new Step(1, 0, (x, Rnd.F(60, 1)), (idx, vals)), atol: 0, rtol: 0, what: $"b={batchDims} q={q}");
    }

    [Theory]
    [InlineData("none", 1)] [InlineData("add", 1)] [InlineData("mul", 0)] [InlineData("max", 1)] [InlineData("min", 0)] [InlineData("add", 0)]
    public void ScatterElements(string red, int axis)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", 4, 6); var idx = tg.In("idx", DataType.I32, 3, 4); var u = tg.InF("u", 3, 4);
        long lim = axis == 1 ? 6 : 4;
        tg.B.Output("y", tg.B.Emit1("ScatterElements", [x, idx, u], Attributes.From(("axis", (long)axis), ("reduction", red))));
        // unique indices for "none" along the axis so the result is order independent
        int[] iv = axis == 1 ? [0, 2, 5, 1, 3, 4, 5, 0, 2, 1, 4, 3] : [1, 3, 0, 2, 3, 1, 0, 2, 0, 1, 2, 3];
        if (red != "none") iv = Rnd.I(12, 7, (int)lim);
        Diff.Check("ScatterElements", tg.B.Build(), new Step(1, 0, (x, Rnd.F(24, 1)), (idx, iv), (u, Rnd.F(12, 2))), atol: 1e-5, what: $"{red} axis {axis}");
    }

    [Theory]
    [InlineData("none")] [InlineData("add")] [InlineData("max")]
    public void ScatterND(string red)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", 4, 5, 3); var idx = tg.In("idx", DataType.I32, 4, 2); var u = tg.InF("u", 4, 3);
        tg.B.Output("y", tg.B.Emit1("ScatterND", [x, idx, u], Attributes.From(("reduction", red))));
        var iv = new[] { 0, 1, 1, 2, 2, 4, 3, 0 };   // unique rows
        if (red != "none") iv = [0, 1, 0, 1, 2, 2, 0, 1];
        Diff.Check("ScatterND", tg.B.Build(), new Step(1, 0, (x, Rnd.F(60, 1)), (idx, iv), (u, Rnd.F(12, 2))), atol: 1e-5, what: red);
    }

    [Theory]
    [InlineData(false, false)] [InlineData(true, false)] [InlineData(false, true)] [InlineData(true, true)]
    public void CumSum(bool exclusive, bool reverse)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 5, 9);
        tg.B.Output("a1", tg.B.Emit1("CumSum", [x], Attributes.From(("axis", 2L), ("exclusive", exclusive), ("reverse", reverse))));
        tg.B.Output("a0", tg.B.Emit1("CumSum", [x], Attributes.From(("axis", 1L), ("exclusive", exclusive), ("reverse", reverse))));
        Diff.Check("CumSum", tg.B.Build(), new Step(3, 0, (x, Rnd.F(135, 1))), atol: 1e-5, what: $"excl={exclusive} rev={reverse}");
    }

    [Theory]
    [InlineData("max", 1, true, false)] [InlineData("min", 1, true, false)] [InlineData("max", 2, false, false)] [InlineData("max", 1, true, true)] [InlineData("min", 2, false, true)]
    public void ArgReduce(string mode, int axis, bool keep, bool last)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 7, 6);
        tg.B.Output("y", tg.B.Emit1("ArgReduce", [x], Attributes.From(("mode", mode), ("axis", (long)axis), ("keepdims", keep), ("select_last_index", last))));
        var data = Rnd.F(3 * 42, 3).Select(v => MathF.Round(v * 3)).ToArray();    // many ties
        Diff.Check("ArgReduce", tg.B.Build(), new Step(3, 0, (x, data)), what: $"{mode} axis {axis} keep={keep} last={last}");
    }

    [Theory]
    [InlineData(true, 1, 3)] [InlineData(false, 1, 4)] [InlineData(true, 2, 5)] [InlineData(true, 1, 7)]
    public void TopK(bool largest, int axis, int k)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 7, 9);
        var outs = tg.B.Emit("TopK", [x], Attributes.From(("k", (long)k), ("axis", (long)axis), ("largest", largest)));
        tg.B.Output("v", outs[0]); tg.B.Output("i", outs[1]);
        var data = Rnd.F(2 * 63, 3).Select(v => MathF.Round(v * 4) / 4).ToArray();   // ties
        Diff.Check("TopK", tg.B.Build(), new Step(2, 0, (x, data)), what: $"largest={largest} axis={axis} k={k}");
    }

    [Theory]
    [InlineData(0, 1, 4)] [InlineData(0, -1, 4)] [InlineData(1, 2, 5)]
    public void Trilu(int k, int unused, int n)
    {
        Gpu.Require();
        foreach (var upper in new[] { true, false })
        {
            var tg = new TG(); var x = tg.InF("x", TG.T, n, n + 2);
            tg.B.Output("y", tg.B.Emit1("Trilu", [x], Attributes.From(("upper", upper), ("k", (long)(k + unused)))));
            Diff.Check("Trilu", tg.B.Build(), new Step(2, 0, (x, Rnd.F(2 * n * (n + 2), 1))), atol: 0, what: $"upper={upper} k={k + unused}");
        }
    }

    [Theory] [InlineData(-1)] [InlineData(1)]
    public void OneHot(int axis)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.In("x", DataType.I32, TG.T, 4);
        tg.B.Output("y", tg.B.Emit1("OneHot", [x], Attributes.From(("depth", 6L), ("axis", (long)axis), ("on", 2f), ("off", -1f))));
        Diff.Check("OneHot", tg.B.Build(), new Step(3, 0, (x, new[] { 0, 5, -1, 3, 2, 2, 4, 1, 0, 0, 5, 3 })), atol: 0, what: $"axis {axis}");
    }
}

public class GeneralNumericTests
{
    [Theory]
    [InlineData("prod")] [InlineData("l1")] [InlineData("l2")] [InlineData("sumsquare")] [InlineData("logsumexp")]
    public void ReduceModes(string mode)
    {
        Gpu.Require();
        foreach (var (axes, keep) in new[] { (new long[] { 2, 3 }, true), (new long[] { 1 }, false) })
        {
            var tg = new TG(); var x = tg.InF("x", TG.T, 6, 5, 4);
            tg.B.Output("y", tg.B.Emit1("Reduce", [x], Attributes.From(("mode", mode), ("axes", axes), ("keepdims", keep))));
            Diff.Check("Reduce." + mode, tg.B.Build(), new Step(2, 0, (x, Rnd.F(240, 1, mode == "prod" ? 1.2f : 3f))), atol: 1e-4, rtol: 1e-4, what: $"axes {string.Join(",", axes)}");
        }
    }

    [Fact]
    public void ReduceLogSumExpLargeAxis()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 5000);
        tg.B.Output("y", tg.B.Emit1("Reduce", [x], Attributes.From(("mode", "logsumexp"), ("axes", new long[] { 1 }), ("keepdims", false))));
        Diff.Check("Reduce.logsumexp.large", tg.B.Build(), new Step(2, 0, (x, Rnd.F(10000, 1, 20f))), atol: 1e-4, rtol: 1e-5);
    }

    [Fact]
    public void LogSoftmax()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 3, 50);
        tg.B.Output("y", tg.B.Emit1("Softmax", [x], Attributes.From(("log", true), ("scale", 0.5f))));
        Diff.Check("Softmax.log", tg.B.Build(), new Step(4, 0, (x, Rnd.F(600, 1, 8f))), atol: 1e-5, rtol: 1e-5);
    }

    [Theory]
    [InlineData(4, 1, 6, true, true)] [InlineData(4, 4, 6, true, true)] [InlineData(12, 3, 5, true, false)] [InlineData(8, 2, 7, false, true)] [InlineData(32, 8, 3, false, false)]
    public void GroupNorm(int c, int groups, int s, bool scale, bool bias)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, c, s, 3);
        var args = new List<Value> { x };
        if (scale) args.Add(tg.RandW("s", DataType.F32, 1f, c));
        if (bias) args.Add(tg.RandW("b", DataType.F32, 1f, c));
        tg.B.Output("y", tg.B.Emit1("GroupNorm", args, Attributes.From(("groups", (long)groups), ("eps", 1e-5f))));
        Diff.Check("GroupNorm", tg.B.Build(), new Step(3, 0, (x, Rnd.F(3 * c * s * 3, 1, 2f))), atol: 5e-5, rtol: 1e-4, what: $"c={c} g={groups}");
    }

    [Theory]
    [InlineData(1, 8, 5, 5, 6, 3, 1, 0, 0, 1, 1)]
    [InlineData(2, 8, 5, 6, 6, 3, 2, 1, 1, 1, 1)]
    [InlineData(1, 4, 4, 4, 4, 4, 2, 1, 0, 1, 1)]
    [InlineData(1, 6, 5, 5, 6, 3, 1, 1, 0, 2, 3)]
    [InlineData(2, 8, 4, 4, 8, 2, 2, 0, 0, 1, 4)]
    [InlineData(1, 4, 5, 7, 4, 3, 3, 2, 2, 1, 1)]
    public void ConvTranspose(int n, int c, int h, int w, int o, int k, int stride, int pad, int outPad, int dil, int groups)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, c, h, w);
        var wt = tg.RandW("w", DataType.F32, 0.4f, c, o / groups, k, k);
        var b = tg.RandW("b", DataType.F32, 1f, o);
        tg.B.Output("y", tg.B.Emit1("ConvTranspose", [x, wt, b], Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad }),
            ("dilations", new long[] { dil, dil }), ("groups", (long)groups), ("output_padding", new long[] { outPad, outPad }))));
        Diff.Check("ConvTranspose", tg.B.Build(), new Step(n, 0, (x, Rnd.F(n * c * h * w, 1))), atol: 1e-4, rtol: 1e-4, what: $"k{k} s{stride} p{pad} op{outPad} d{dil} g{groups}");
    }

    [Fact]
    public void ConvTransposeNoBiasF16Weights()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 6, 5, 5);
        var wt = tg.RandW("w", DataType.F16, 0.4f, 6, 4, 3, 3);
        tg.B.Output("y", tg.B.Emit1("ConvTranspose", [x, wt], Attributes.From(("strides", new long[] { 2, 2 }), ("pads", new long[] { 1, 1, 1, 1 }))));
        Diff.Check("ConvTranspose.f16", tg.B.Build(), new Step(2, 0, (x, Rnd.F(300, 1))), atol: 1e-4, rtol: 1e-4);
    }

    [Theory]
    [InlineData("linear", "zeros", false)] [InlineData("linear", "zeros", true)] [InlineData("linear", "border", false)] [InlineData("linear", "reflection", true)]
    [InlineData("linear", "reflection", false)] [InlineData("nearest", "zeros", false)] [InlineData("nearest", "border", true)] [InlineData("cubic", "zeros", false)] [InlineData("cubic", "border", true)]
    public void GridSample(string mode, string padding, bool align)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 3, 6, 7); var grid = tg.InF("g", TG.T, 4, 5, 2);
        tg.B.Output("y", tg.B.Emit1("GridSample", [x, grid], Attributes.From(("mode", mode), ("padding", padding), ("align_corners", align))));
        Diff.Check("GridSample", tg.B.Build(), new Step(2, 0, (x, Rnd.F(2 * 126, 1)), (grid, Rnd.F(2 * 40, 2, 1.4f))), atol: 2e-4, rtol: 1e-4, what: $"{mode}/{padding}/align={align}");
    }
}
