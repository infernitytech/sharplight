using Sharplight.Models;
using Xunit;

namespace Sharplight.Import.Onnx.Tests;

/// <summary>
/// Attention written out with elementary ops (MatMul → scale → [mask Add] → Softmax → MatMul) becomes one fused Attention op, and
/// computes what the elementary form computes, on the CPU and (when present) Vulkan.
/// </summary>
public sealed class AttentionPatternTests
{
    private const int S = 10;
    private const int H = 2;
    private const int D = 16;
    private const int W = H * D;

    private static IEnumerable<string> Ops(ModelDefinition d) => d.Graph("main").Nodes.Select(n => n.Op.Name);

    /// <summary>Rank-4 heads (vit / bert exports): Kᵀ by a [0, 2, 3, 1] transpose, a Div scale and an optional key-padding mask.</summary>
    private static TestModel Rank4(bool mask, bool fixedBatch = false)
    {
        object batch = fixedBatch ? 3L : "N";
        var m = new TestModel().Opset(13).Input("x", batch, S, W);
        if (mask)
        {
            m.Input("m", batch, 1, 1, S);
        }

        m.Output("y", batch, S, W)
            .InitF32("wq", [W, W], Rnd(W * W, 1)).InitF32("wk", [W, W], Rnd(W * W, 2)).InitF32("wv", [W, W], Rnd(W * W, 3))
            .InitI64("heads", [4], 0, S, H, D).InitI64("merged", [3], 0, S, W).InitF32("c", [], [4f])
            .Node("MatMul", ["x", "wq"], ["q0"]).Node("Reshape", ["q0", "heads"], ["q1"]).Node("Transpose", ["q1"], ["q"], TestModel.AttrInts("perm", 0, 2, 1, 3))
            .Node("MatMul", ["x", "wk"], ["k0"]).Node("Reshape", ["k0", "heads"], ["k1"]).Node("Transpose", ["k1"], ["kt"], TestModel.AttrInts("perm", 0, 2, 3, 1))
            .Node("MatMul", ["x", "wv"], ["v0"]).Node("Reshape", ["v0", "heads"], ["v1"]).Node("Transpose", ["v1"], ["v"], TestModel.AttrInts("perm", 0, 2, 1, 3))
            .Node("MatMul", ["q", "kt"], ["s0"]).Node("Div", ["s0", "c"], ["s1"]);
        if (mask)
        {
            m.Node("Add", ["s1", "m"], ["s2"]).Node("Softmax", ["s2"], ["p"], TestModel.Attr("axis", -1));
        }
        else
        {
            m.Node("Softmax", ["s1"], ["p"], TestModel.Attr("axis", -1));
        }

        return m.Node("MatMul", ["p", "v"], ["o0"]).Node("Transpose", ["o0"], ["o1"], TestModel.AttrInts("perm", 0, 2, 1, 3))
            .Node("Reshape", ["o1", "merged"], ["y"]);
    }

    /// <summary>
    /// Rank-3 heads [B·H, S, D] (whisper exports): Q pre-scaled, Kᵀ by a [0, 2, 1] transpose of the merged keys. With
    /// <paramref name="roundTrip"/> the probabilities go through Reshapes to [B, H, S, S] and back, as in exports that can also
    /// return them.
    /// </summary>
    private static TestModel Rank3(bool roundTrip = false)
    {
        var m = new TestModel().Opset(13).Input("x", 2L, S, W).Output("y", 2L, S, W)
            .InitF32("wq", [W, W], Rnd(W * W, 4)).InitF32("wk", [W, W], Rnd(W * W, 5)).InitF32("wv", [W, W], Rnd(W * W, 6))
            .InitI64("heads", [4], 2, S, H, D).InitI64("rows", [3], 2 * H, S, D).InitI64("merged", [3], 2, S, W).InitF32("c", [], [0.25f])
            .Node("MatMul", ["x", "wq"], ["q0"]).Node("Mul", ["q0", "c"], ["qs"]).Node("Reshape", ["qs", "heads"], ["q1"])
            .Node("Transpose", ["q1"], ["q2"], TestModel.AttrInts("perm", 0, 2, 1, 3)).Node("Reshape", ["q2", "rows"], ["q"])
            .Node("MatMul", ["x", "wk"], ["k0"]).Node("Reshape", ["k0", "heads"], ["k1"])
            .Node("Transpose", ["k1"], ["k2"], TestModel.AttrInts("perm", 0, 2, 1, 3)).Node("Reshape", ["k2", "rows"], ["k"])
            .Node("Transpose", ["k"], ["kt"], TestModel.AttrInts("perm", 0, 2, 1))
            .Node("MatMul", ["x", "wv"], ["v0"]).Node("Reshape", ["v0", "heads"], ["v1"])
            .Node("Transpose", ["v1"], ["v2"], TestModel.AttrInts("perm", 0, 2, 1, 3)).Node("Reshape", ["v2", "rows"], ["v"])
            .InitI64("probs4", [4], 2, H, S, S).InitI64("probs3", [3], 2 * H, S, S)
            .Node("MatMul", ["q", "kt"], ["s0"]).Node("Softmax", ["s0"], [roundTrip ? "p0" : "p"], TestModel.Attr("axis", -1));
        if (roundTrip)
        {
            m.Node("Reshape", ["p0", "probs4"], ["p4"]).Node("Reshape", ["p4", "probs3"], ["p"]);
        }

        return m.Node("MatMul", ["p", "v"], ["o0"]).Node("Reshape", ["o0", "heads"], ["o1"])
            .Node("Transpose", ["o1"], ["o2"], TestModel.AttrInts("perm", 0, 2, 1, 3)).Node("Reshape", ["o2", "merged"], ["y"]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Rank4_IsFused(bool mask)
    {
        var ops = Ops(Rank4(mask).Build()).ToList();
        Assert.Contains("Attention", ops);
        Assert.DoesNotContain("Softmax", ops);
        Assert.DoesNotContain("BatchMatMul", ops);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Rank3_IsFused(bool roundTrip)
    {
        var ops = Ops(Rank3(roundTrip).Build()).ToList();
        Assert.Contains("Attention", ops);
        Assert.DoesNotContain("Softmax", ops);
    }

    [Fact]
    public void FusionOff_KeepsElementaryOps()
    {
        var ops = Ops(Rank4(mask: true).Build(new OnnxImportOptions { FuseAttention = false })).ToList();
        Assert.DoesNotContain("Attention", ops);
        Assert.Contains("Softmax", ops);
    }

    /// <summary>Probabilities that are also a graph output keep the elementary form (they must be computed anyway).</summary>
    [Fact]
    public void ProbabilitiesWithSecondReader_AreNotFused()
    {
        var model = Rank4(mask: false);
        model.Output("p", "N", H, S, S);
        Assert.DoesNotContain("Attention", Ops(model.Build()));
    }

    /// <summary>The fused form computes what the elementary ops compute (online softmax: f32 rounding only).</summary>
    [Theory]
    [InlineData("cpu", 0)]
    [InlineData("cpu", 1)]
    [InlineData("cpu", 2)]
    [InlineData("cpu", 3)]
    [InlineData("vulkan", 0)]
    [InlineData("vulkan", 1)]
    [InlineData("vulkan", 2)]
    [InlineData("vulkan", 3)]
    public void FusedMatchesElementary(string provider, int form)
    {
        var device = Providers.Get(provider);
        var model = form switch { 0 => Rank4(mask: false), 1 => Rank4(mask: true, fixedBatch: true), 2 => Rank3(), _ => Rank3(roundTrip: true) };
        int batch = form >= 2 ? 2 : 3;
        List<float[]> inputs = [Rnd(batch * S * W, 11)];
        if (form == 1)
        {
            inputs.Add([.. Enumerable.Range(0, batch * S).Select(i => i % 4 == 3 ? -10000f : 0f)]);
        }

        string path = Path.Combine(Path.GetTempPath(), $"sl-attn-pattern-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(path, model.ToBytes());
        try
        {
            var fused = OnnxExec.Run(path, new OnnxImportOptions(), device, batch, inputs);
            var plain = OnnxExec.Run(path, new OnnxImportOptions { FuseAttention = false }, device, batch, inputs);
            var a = fused.Outputs[0];
            var b = plain.Outputs[0];
            Assert.Equal(b.Length, a.Length);
            for (int i = 0; i < a.Length; i++)
            {
                Assert.True(Math.Abs(a[i] - b[i]) <= 1e-5 + 1e-5 * Math.Abs(b[i]), $"[{i}] fused {a[i]} vs elementary {b[i]}");
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static float[] Rnd(int n, int seed)
    {
        var r = new Random(seed);
        return [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1) * 0.5f)];
    }
}
