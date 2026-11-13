using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class ReduceTests
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
    [InlineData("sum")]
    [InlineData("mean")]
    [InlineData("max")]
    [InlineData("min")]
    public void GlobalAveragePoolStyle(string mode) => Run("hw " + mode, mode, [2, 3], true, [24, 7, 7]);

    [Theory]
    [InlineData("sum")]
    [InlineData("mean")]
    [InlineData("max")]
    [InlineData("min")]
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
        var tg = new TG();
        var x = tg.InF("x", 4, 9, 11);
        tg.B.Output("y", tg.B.Emit1("Reduce", [x], Attributes.From(("mode", "sum"), ("axes", new long[] { 0, 1, 2 }), ("keepdims", false))));
        Diff.Check("Reduce.fixed", tg.B.Build(), new Step(1, 0, (x, Rnd.F(4 * 99, 1))), atol: 1e-3);
    }
}
