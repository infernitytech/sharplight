using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class BatchMatMulTests
{
    private static void Run(string label, int[] aShape, int[] bShape, bool ta, bool tb, float alpha = 1f, int sym = 0, int[]? tvals = null)
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

    // Long K over few output tiles: K split across workgroups (partials summed by bmm_reduce, alpha applied there).
    [Fact] public void SplitK() => Run("split-k", [2, 70, 1100], [2, 1100, 40], false, false, alpha: 0.5f);
    [Fact] public void SplitKSymbolicBatch() => Run("split-k sym", [1, 30, 600], [1, 20, 600], false, true, sym: 1, tvals: [1, 3]);
    [Fact] public void SplitKSymbolicRows() => Run("split-k sym M", [1, 700], [700, 9], false, false, sym: 1, tvals: [1, 5, 30]);
}
