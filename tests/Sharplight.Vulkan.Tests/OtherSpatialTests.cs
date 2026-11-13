using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public class PoolTests
{
    private static void Run(string label, int n, int c, int h, int w, string mode, int[] k, int[] s, int[] pads, int[] dil, bool ceil, bool countPad, bool slice = false)
    {
        Gpu.Require();
        var tg = new TG();
        int cin = slice ? c + 3 : c;
        var x = tg.InF("x", TG.T, cin, h, w);
        var xv = slice ? tg.B.Slice(x, 1, 1, c) : x;
        var attrs = Attributes.From(("mode", mode), ("kernel", k.Select(v => (long)v).ToArray()), ("strides", s.Select(v => (long)v).ToArray()),
            ("pads", pads.Select(v => (long)v).ToArray()), ("dilations", dil.Select(v => (long)v).ToArray()), ("ceil_mode", ceil), ("count_include_pad", countPad));
        tg.B.Output("y", tg.B.Emit1("Pool", [xv], attrs));
        Diff.Check("Pool." + mode, tg.B.Build(), new Step(n, 0, (x, Rnd.F((long)n * cin * h * w, 3))), what: label);
    }

    [Theory]
    [InlineData("max", 2, 8, 15, 15, 3, 2, 1, false, false)]
    [InlineData("max", 1, 8, 16, 16, 2, 2, 0, false, false)]
    [InlineData("max", 1, 8, 15, 15, 3, 2, 0, true, false)]
    [InlineData("avg", 2, 8, 15, 15, 3, 2, 1, false, false)]
    [InlineData("avg", 2, 8, 15, 15, 3, 2, 1, false, true)]
    [InlineData("avg", 1, 8, 14, 14, 2, 2, 0, true, true)]
    [InlineData("avg", 1, 8, 15, 15, 3, 2, 1, true, false)]
    [InlineData("avg", 1, 8, 15, 15, 3, 2, 1, true, true)]
    [InlineData("max", 1, 4, 7, 7, 7, 1, 0, false, false)]
    public void Square(string mode, int n, int c, int h, int w, int k, int s, int p, bool ceil, bool cp)
        => Run($"{mode} k{k} s{s} p{p} ceil{ceil} cp{cp}", n, c, h, w, mode, [k, k], [s, s], [p, p, p, p], [1, 1], ceil, cp);

    [Fact] public void MaxDilated() => Run("max dil2", 1, 6, 17, 17, "max", [3, 3], [1, 1], [2, 2, 2, 2], [2, 2], false, false);
    [Fact] public void MaxAsymmetric() => Run("max asym", 2, 5, 13, 17, "max", [3, 2], [2, 1], [1, 0, 2, 1], [1, 1], false, false);
    [Fact] public void AvgAsymmetricCountPad() => Run("avg asym cp", 2, 5, 13, 17, "avg", [3, 2], [2, 1], [1, 0, 2, 1], [1, 1], false, true);
    [Fact] public void AvgAsymmetricCeil() => Run("avg asym ceil", 2, 5, 13, 17, "avg", [3, 2], [2, 2], [1, 0, 1, 1], [1, 1], true, true);
    [Fact] public void StridedInput() => Run("sliced", 2, 6, 11, 11, "max", [3, 3], [2, 2], [1, 1, 1, 1], [1, 1], false, false, slice: true);
}

public class ReduceTests
{
    private static void Run(string label, string mode, long[] axes, bool keep, int[] shape, int n = 3, bool permute = false)
    {
        Gpu.Require();
        var tg = new TG();
        var dims = shape.Select(d => (Dim)d).ToArray();
        var x = tg.InF("x", [TG.T, .. dims]);
        var xv = x;
        long total = n; foreach (var d in shape)
        {
            total *= d;
        }

        if (permute)
        {
            xv = tg.B.Permute(x, [0, .. Enumerable.Range(1, shape.Length).Reverse()]);
        }
        // user axes refer to the (possibly permuted) value, rank = shape.Length + 1
        tg.B.Output("y", tg.B.Emit1("Reduce", [xv], Attributes.From(("mode", mode), ("axes", axes), ("keepdims", keep))));
        Diff.Check("Reduce." + mode, tg.B.Build(), new Step(n, 0, (x, Rnd.F(total, 5))), atol: 2e-4, rtol: 1e-4, what: label);
    }

    [Theory]
    [InlineData("sum")] [InlineData("mean")] [InlineData("max")] [InlineData("min")]
    public void GlobalAveragePoolStyle(string mode) => Run("hw " + mode, mode, [2, 3], true, [24, 7, 7]);

    [Theory]
    [InlineData("sum")] [InlineData("mean")] [InlineData("max")] [InlineData("min")]
    public void ChannelAxis(string mode) => Run("c " + mode, mode, [1], false, [24, 5, 6]);

    [Fact] public void LastAxis() => Run("last", "mean", [-1], true, [10, 131]);
    [Fact] public void AllButBatch() => Run("all", "sum", [1, 2, 3], false, [8, 9, 10]);
    [Fact] public void MiddleAxis() => Run("mid", "max", [2], true, [6, 11, 7]);
    [Fact] public void NonAdjacentAxes() => Run("nonadj", "sum", [1, 3], true, [6, 5, 7]);
    [Fact] public void LargeReduction() => Run("large", "sum", [1], false, [20000], n: 4);
    [Fact] public void LargeReductionMax() => Run("large max", "max", [1, 2], false, [300, 100], n: 2);
    [Fact] public void PermutedInput() => Run("permuted", "mean", [1], false, [6, 7, 8], permute: true);
    [Fact] public void PermutedInputKeep() => Run("permuted2", "sum", [2, 3], true, [6, 7, 8], permute: true);
    [Fact] public void ScalarLikeResult() => Run("axes all but T", "max", [1], true, [1000]);

    [Fact]
    public void FixedBatchFullReduce()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", 4, 9, 11);
        tg.B.Output("y", tg.B.Emit1("Reduce", [x], Attributes.From(("mode", "sum"), ("axes", new long[] { 0, 1, 2 }), ("keepdims", false))));
        Diff.Check("Reduce.fixed", tg.B.Build(), new Step(1, 0, (x, Rnd.F(4 * 99, 1))), atol: 1e-3);
    }
}

public class BatchMatMulTests
{
    private static void Run(string label, int[] aShape, int[] bShape, bool ta, bool tb, float alpha = 1f, int sym = 0, bool permuteA = false, bool permuteB = false, int[]? tvals = null)
    {
        Gpu.Require();
        var tg = new TG();
        Dim[] Dims(int[] s) => [.. s.Select((d, i) => i == 0 && sym == 1 ? TG.T : (Dim)d)];
        var a = tg.InF("a", Dims(aShape));
        Dim[] bd = bShape.Select((d, i) => (Dim)d).ToArray();
        if (sym == 2)
        {
            bd[0] = TG.T;
        }

        var b = tg.InF("b", bd);
        tg.B.Output("y", tg.B.Emit1("BatchMatMul", [a, b], Attributes.From(("trans_a", ta), ("trans_b", tb), ("alpha", alpha))));
        var g = tg.B.Build();
        var steps = new List<Step>();
        foreach (var t in tvals ?? [aShape[0]])
        {
            long na = aShape.Select((d, i) => i == 0 && sym == 1 ? t : d).Aggregate(1L, (p, q) => p * q);
            long nb = bShape.Select((d, i) => i == 0 && sym == 2 ? t : d).Aggregate(1L, (p, q) => p * q);
            steps.Add(new Step(t, 0, (a, Rnd.F(na, 1 + (int)t)), (b, Rnd.F(nb, 2 + (int)t))));
        }
        Diff.Check("BatchMatMul", g, steps, atol: 2e-4, rtol: 2e-4, maxTokens: 32, what: label);
    }

    [Fact] public void Plain3D() => Run("3d", [4, 17, 33], [4, 33, 21], false, false);
    [Fact] public void TransB() => Run("transB", [4, 17, 33], [4, 21, 33], false, true);
    [Fact] public void TransA() => Run("transA", [4, 33, 17], [4, 33, 21], true, false);
    [Fact] public void TransBoth() => Run("transAB", [4, 33, 17], [4, 21, 33], true, true, alpha: 0.125f);
    [Fact] public void Rank2() => Run("2d", [37, 29], [29, 41], false, false);
    [Fact] public void BroadcastBatchB() => Run("bcast b", [5, 12, 16], [16, 20], false, false);
    [Fact] public void BroadcastBatchA() => Run("bcast a", [12, 16], [5, 16, 20], false, false);
    [Fact] public void Rank4Attention() => Run("attn qk", [2, 4, 25, 16], [2, 4, 25, 16], false, true, alpha: 0.25f);
    [Fact] public void Rank4BroadcastMiddle() => Run("bcast mid", [2, 4, 9, 8], [2, 1, 8, 11], false, false);
    [Fact] public void Rank4BroadcastOuter() => Run("bcast outer", [1, 4, 9, 8], [3, 4, 8, 11], false, false);
    [Fact] public void BigTileCount() => Run("big", [3, 130, 70], [3, 70, 150], false, false);
    [Fact] public void SmallK() => Run("k=1", [6, 8, 1], [6, 1, 9], false, false);
    [Fact] public void SymbolicBatchDim() => Run("sym batch", [1, 12, 20], [1, 20, 14], false, false, sym: 1, tvals: [1, 3, 7]);
    [Fact] public void SymbolicBatchDim4D() => Run("sym batch 4d", [1, 4, 11, 8], [1, 4, 13, 8], false, true, sym: 1, tvals: [2, 5]);
    [Fact] public void SymbolicRowsRank2() => Run("sym M", [1, 24], [24, 17], false, false, sym: 1, tvals: [1, 9, 31]);
    [Fact] public void SymbolicBBroadcastA() => Run("sym b", [4, 6], [1, 6, 5], false, false, sym: 2, tvals: [3]);
}

public class PadTests
{
    private static void Run(string label, string mode, int[] shape, long[] pads, float value = 0f, bool slice = false)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", [TG.T, .. shape.Select(d => (Dim)d)]);
        var xv = x;
        if (slice)
        {
            xv = tg.B.Slice(x, 2, 1, shape[1] - 2);
        }

        tg.B.Output("y", tg.B.Emit1("Pad", [xv], Attributes.From(("pads", pads), ("mode", mode), ("value", value))));
        long n = 3; foreach (var d in shape)
        {
            n *= d;
        }

        Diff.Check("Pad." + mode, tg.B.Build(), new Step(3, 0, (x, Rnd.F(n, 2))), what: label);
    }

    // pads layout: begin per axis (incl. the batch axis), then end per axis
    [Theory]
    [InlineData("constant")] [InlineData("reflect")] [InlineData("edge")]
    public void Spatial4D(string mode) => Run(mode, mode, [4, 9, 11], [0, 0, 2, 1, 0, 0, 3, 2], 1.5f);

    [Theory]
    [InlineData("constant")] [InlineData("reflect")] [InlineData("edge")]
    public void Rank3(string mode) => Run(mode, mode, [10, 7], [0, 1, 2, 0, 3, 1], -0.5f);

    [Fact] public void ConstantOnlyOneSide() => Run("one side", "constant", [3, 5, 6], [0, 0, 0, 2, 0, 0, 0, 0], 7f);
    [Fact] public void ConstantNegativeCrop() => Run("crop", "constant", [3, 8, 9], [0, 0, -1, 2, 0, 0, -2, 1]);
    [Fact] public void ReflectLargePad() => Run("reflect large", "reflect", [3, 4, 5], [0, 0, 0, 0, 0, 0, 5, 7]);
    [Fact] public void EdgeOnStridedInput() => Run("edge strided", "edge", [4, 9, 8], [0, 0, 1, 2, 0, 0, 2, 3], slice: true);
    [Fact] public void ReflectOnStridedInput() => Run("reflect strided", "reflect", [4, 9, 8], [0, 0, 1, 2, 0, 0, 2, 3], slice: true);
}

public class ResizeTests
{
    private static void Run(string label, string mode, string coord, string nearest, int n, int c, int h, int w, int oh, int ow, bool slice = false)
    {
        Gpu.Require();
        var tg = new TG();
        int cin = slice ? c + 2 : c;
        var x = tg.InF("x", TG.T, cin, h, w);
        var xv = slice ? tg.B.Slice(x, 1, 1, c) : x;
        tg.B.Output("y", tg.B.Emit1("Resize", [xv], Attributes.From(("size", new long[] { oh, ow }), ("mode", mode), ("coord", coord), ("nearest_mode", nearest))));
        Diff.Check("Resize." + mode, tg.B.Build(), new Step(n, 0, (x, Rnd.F((long)n * cin * h * w, 4))), atol: 2e-5, rtol: 2e-5, what: label);
    }

    public static IEnumerable<object[]> NearestCases()
    {
        foreach (var coord in new[] { "half_pixel", "asymmetric", "align_corners", "pytorch_half_pixel" })
        {
            foreach (var nm in new[] { "round_prefer_floor", "round_prefer_ceil", "floor", "ceil" })
            {
                foreach (var (oh, ow) in new[] { (16, 14), (5, 6), (7, 9) })
                {
                    yield return [coord, nm, oh, ow];
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(NearestCases))]
    public void Nearest(string coord, string nm, int oh, int ow) => Run($"{coord}/{nm} {oh}x{ow}", "nearest", coord, nm, 2, 3, 8, 7, oh, ow);

    public static IEnumerable<object[]> LinearCases()
    {
        foreach (var coord in new[] { "half_pixel", "asymmetric", "align_corners", "pytorch_half_pixel" })
        {
            foreach (var (oh, ow) in new[] { (16, 14), (5, 6), (7, 9), (1, 1), (8, 7) })
            {
                yield return [coord, oh, ow];
            }
        }
    }

    [Theory]
    [MemberData(nameof(LinearCases))]
    public void Linear(string coord, int oh, int ow) => Run($"{coord} {oh}x{ow}", "linear", coord, "round_prefer_floor", 2, 3, 8, 7, oh, ow);

    [Fact] public void Upsample2x() => Run("2x", "nearest", "asymmetric", "floor", 1, 16, 14, 14, 28, 28);
    [Fact] public void BilinearUpsample2xStrided() => Run("2x bilinear sliced", "linear", "half_pixel", "round_prefer_floor", 2, 4, 10, 10, 20, 20, slice: true);
}

public class RowOpTests
{
    [Theory]
    [InlineData(1, 100, 1f)] [InlineData(7, 1000, 0.125f)] [InlineData(5, 33, 1f)] [InlineData(3, 8, 2f)] [InlineData(2, 5000, 1f)]
    public void Softmax(int t, int d, float scale)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, d);
        tg.B.Output("y", tg.B.Emit1("Softmax", [x], Attributes.From(("scale", scale))));
        Diff.Check("Softmax", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * d, 3, 6f))), atol: 1e-6, rtol: 1e-4, maxTokens: 8);
    }

    [Fact]
    public void Softmax4DAndStrided()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 4, 9, 20);
        var s = tg.B.Slice(x, 3, 2, 13);   // strided last axis slice (stride 20, inner contiguous)
        tg.B.Output("y", tg.B.Emit1("Softmax", [s], Attributes.Empty));
        Diff.Check("Softmax.4d", tg.B.Build(), new Step(3, 0, (x, Rnd.F(3 * 4 * 9 * 20, 3, 3f))), atol: 1e-6);
    }

    [Fact]
    public void SoftmaxOnPermutedInputFallsBackToDense()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 6, 10);
        var p = tg.B.Permute(x, 0, 2, 1);   // inner axis no longer contiguous
        tg.B.Output("y", tg.B.Emit1("Softmax", [p], Attributes.Empty));
        Diff.Check("Softmax.permuted", tg.B.Build(), new Step(3, 0, (x, Rnd.F(3 * 60, 3, 3f))), atol: 1e-6);
    }

    [Theory]
    [InlineData(3, 64, true, true)] [InlineData(5, 768, true, true)] [InlineData(4, 100, false, false)] [InlineData(4, 100, true, false)] [InlineData(2, 33, false, true)]
    public void LayerNorm(int t, int d, bool w, bool b)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 5, d);
        var args = new List<Value> { x };
        if (w)
        {
            args.Add(tg.RandW("w", DataType.F32, 1f, d));
        }

        if (b)
        {
            args.Add(tg.RandW("b", DataType.F32, 1f, d));
        }

        tg.B.Output("y", tg.B.Emit1("LayerNorm", args, Attributes.From(("eps", 1e-5f))));
        Diff.Check("LayerNorm", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * 5 * d, 3, 3f))), atol: 2e-5, rtol: 1e-4);
    }

    [Fact]
    public void LayerNormHalfWeights()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 48);
        var w = tg.RandW("w", DataType.F16, 1f, 48); var b = tg.RandW("b", DataType.BF16, 1f, 48);
        tg.B.Output("y", tg.B.Emit1("LayerNorm", [x, w, b], Attributes.From(("eps", 1e-6f))));
        Diff.Check("LayerNorm.half", tg.B.Build(), new Step(4, 0, (x, Rnd.F(4 * 48, 3, 3f))), atol: 2e-5);
    }

    [Theory]
    [InlineData("ReduceSum")] [InlineData("ReduceMax")] [InlineData("ArgMax")]
    public void RowReduce(string op)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 3, 77);
        tg.B.Output("y", tg.B.Emit1(op, [x], Attributes.Empty));
        Diff.Check(op, tg.B.Build(), new Step(5, 0, (x, Rnd.F(5 * 3 * 77, 3))), atol: 2e-4);
    }

    [Fact]
    public void ArgMaxTieBreaksToFirst()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 40);
        tg.B.Output("y", tg.B.Emit1("ArgMax", [x], Attributes.Empty));
        var data = new float[2 * 40];
        data[7] = 5; data[33] = 5; data[40 + 39] = 2; data[40 + 1] = 2;
        Diff.Check("ArgMax.ties", tg.B.Build(), new Step(2, 0, (x, data)));
    }
}
