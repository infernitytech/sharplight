using Sharplight.Graph;
using Sharplight.Models;
using Xunit;

namespace Sharplight.Onnx.Tests;

/// <summary>Graph-level checks of the ONNX → IR translation (no execution).</summary>
public class TranslationTests
{
    private static float[] Rnd(int n, int seed = 1)
    {
        var r = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() - 0.5)).ToArray();
    }
    private static IEnumerable<string> Ops(ModelDefinition d) => d.Graph("main").Nodes.Select(n => n.Op.Name);

    [Fact]
    public void BatchDim_BecomesSymbolN_AndOutputsKeepNames()
    {
        var def = new TestModel().Input("x", "batch_size", 4).Output("y", "batch_size", 4).Node("Relu", ["x"], ["y"]).Build();
        var g = def.Graph("main");
        Assert.Equal("N", g.Inputs[0].Shape[0].Symbol);
        Assert.Equal(4, g.Inputs[0].Shape[1].Fixed);
        Assert.Equal("y", g.Outputs.Keys.Single());
        Assert.Equal("N", def.Properties[OnnxGraphAdapter.BatchSymbolProperty]);
    }

    [Fact]
    public void DimZero_IsUnknownBatch_AndFixedBatchStaysFixed()
    {
        var def = new TestModel().Input("x", 0, 4).Output("y", 0, 4).Node("Relu", ["x"], ["y"]).Build();
        Assert.True(def.Graph("main").Inputs[0].Shape[0].IsSymbolic);
        var fixedB = new TestModel().Input("x", 2, 4).Output("y", 2, 4).Node("Relu", ["x"], ["y"]).Build();
        Assert.Equal(2, fixedB.Graph("main").Inputs[0].Shape[0].Fixed);
        var forced = new TestModel().Input("x", 2, 4).Output("y", 2, 4).Node("Relu", ["x"], ["y"]).Build(new OnnxImportOptions { ForceSymbolicBatch = true });
        Assert.True(forced.Graph("main").Inputs[0].Shape[0].IsSymbolic);
    }

    [Fact]
    public void DynamicSpatialDim_Errors_UnlessOverridden()
    {
        var m = new TestModel().Input("x", "N", 3, "h", "w").Output("y", "N", 3, "h", "w").Node("Relu", ["x"], ["y"]);
        var e = Assert.Throws<NotSupportedException>(() => m.Build());
        Assert.Contains("dynamic dimension 2", e.Message);
        var def = m.Build(new OnnxImportOptions { InputShapes = { ["x"] = [-1, 3, 32, 48] } });
        var s = def.Graph("main").Inputs[0].Shape;
        Assert.Equal("[N, 3, 32, 48]", s.ToString());
    }

    [Fact]
    public void Image_Input_GetsMediaRole()
    {
        var def = new TestModel().Input("x", "N", 3, 8, 8).Output("y", "N", 3, 8, 8).Node("Relu", ["x"], ["y"]).Build();
        Assert.Equal(InputRole.Media, def.Graph("main").Inputs[0].Input!.Role);
    }

    [Fact]
    public void UnsupportedOps_AreAllListedInOneError()
    {
        var m = new TestModel().Input("x", "N", 4).Output("y", "N", 4)
            .Node("Loop", ["x"], ["a"]).Node("NonMaxSuppression", ["a"], ["b"]).Node("Unique", ["b"], ["c"]).Node("Relu", ["c"], ["y"]);
        var e = Assert.Throws<NotSupportedException>(() => m.Build());
        Assert.Contains("Loop", e.Message);
        Assert.Contains("NonMaxSuppression", e.Message);
        Assert.Contains("Unique", e.Message);
        Assert.DoesNotContain("Relu", e.Message);
    }

    [Fact]
    public void UnusedUnsupportedNode_IsIgnored()
    {
        var def = new TestModel().Input("x", "N", 4).Output("y", "N", 4).Node("Loop", ["x"], ["dead"]).Node("Relu", ["x"], ["y"]).Build();
        Assert.Equal(["Relu"], Ops(def));
    }

    [Fact]
    public void ConstantFolding_ShapeGatherConcatReshape_EmitsOnlyReshape()
    {
        var def = new TestModel().Input("x", "N", 3, 4).Output("y", "N", 12)
            .InitI64("zero", [], 0).InitI64("ax0", [1], 0).InitI64("tail", [1], 12)
            .Node("Shape", ["x"], ["s"]).Node("Gather", ["s", "zero"], ["b"], TestModel.Attr("axis", 0))
            .Node("Unsqueeze", ["b", "ax0"], ["b1"]).Node("Concat", ["b1", "tail"], ["tgt"], TestModel.Attr("axis", 0))
            .Node("Reshape", ["x", "tgt"], ["y"]).Build();
        var nodes = def.Graph("main").Nodes;
        var r = Assert.Single(nodes);
        Assert.Equal("Reshape", r.Op.Name);
        Assert.Equal([0L, 12], r.Attributes.Get<long[]>("shape"));   // batch size became "keep symbolic outer dim"
    }

    [Fact]
    public void ConstantFolding_ArithmeticOnShapes_AndConstantNode()
    {
        var def = new TestModel().Input("x", 2, 3, 4).Output("y", 2, 12)
            .Node("Shape", ["x"], ["s"]).InitI64("idx", [2], 1, 2).Node("Gather", ["s", "idx"], ["hw"])
            .Node("ReduceProd", ["hw"], ["p"], TestModel.Attr("keepdims", 1))
            .InitI64("two", [1], 2).Node("Mul", ["p", "one"], ["pp"]).InitI64("one", [1], 1)
            .Node("Concat", ["two", "pp"], ["tgt"], TestModel.Attr("axis", 0))
            .Node("Reshape", ["x", "tgt"], ["y"]).Build();
        Assert.Equal([2L, 12], Assert.Single(def.Graph("main").Nodes).Attributes.Get<long[]>("shape"));
    }

    [Fact]
    public void BatchSizeArithmetic_IsRejectedWithClearError()
    {
        var m = new TestModel().Input("x", "N", 4).Output("y", "N", 4)
            .InitI64("zero", [], 0).InitI64("two", [], 2)
            .Node("Shape", ["x"], ["s"]).Node("Gather", ["s", "zero"], ["b"]).Node("Mul", ["b", "two"], ["b2"])
            .InitI64("ax", [1], 0).Node("Unsqueeze", ["b2", "ax"], ["b3"]).InitI64("c4", [1], 4).Node("Concat", ["b3", "c4"], ["t"], TestModel.Attr("axis", 0))
            .Node("Reshape", ["x", "t"], ["y"]);
        var e = Assert.Throws<NotSupportedException>(() => m.Build());
        Assert.Contains("symbolic batch", e.Message);
    }

    [Fact]
    public void ConvBatchNorm_IsFoldedIntoDerivedWeights()
    {
        var def = new TestModel().Input("x", "N", 3, 8, 8).Output("y", "N", 4, 6, 6)
            .InitF32("w", [4, 3, 3, 3], Rnd(108)).InitF32("s", [4], [1, 2, 3, 4]).InitF32("b", [4], [0, 1, 0, 1]).InitF32("m", [4], [0, 0, 1, 1]).InitF32("v", [4], [1, 1, 4, 4])
            .Node("Conv", ["x", "w"], ["c"]).Node("BatchNormalization", ["c", "s", "b", "m", "v"], ["y"], TestModel.AttrF("epsilon", 0f)).Build();
        var g = def.Graph("main");
        var conv = Assert.Single(g.Nodes);
        Assert.Equal("Conv", conv.Op.Name);
        Assert.Equal(3, conv.Inputs.Length);   // weight + folded bias
        Assert.Contains(def.Weights.Keys, k => k.EndsWith(".bnfold"));   // derived conv weight
        // bias' = beta - mean * scale/sqrt(var)
        var f = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(conv.Inputs[2].ConstantData!).ToArray();
        Assert.Equal([0f, 1f, -1.5f, -1f], f, new FloatCmp());
    }

    private sealed class FloatCmp : IEqualityComparer<float>
    {
        public bool Equals(float a, float b) => Math.Abs(a - b) < 1e-5;
        public int GetHashCode(float f) => 0;
    }

    [Fact]
    public void BatchNorm_WithSharedConvOutput_IsNotFolded()
    {
        var def = new TestModel().Input("x", "N", 3, 8, 8).Output("y", "N", 4, 6, 6)
            .InitF32("w", [4, 3, 3, 3], Rnd(108)).InitF32("s", [4], [1, 1, 1, 1]).InitF32("b", [4], [0, 0, 0, 0]).InitF32("m", [4], [0, 0, 0, 0]).InitF32("v", [4], [1, 1, 1, 1])
            .Node("Conv", ["x", "w"], ["c"]).Node("BatchNormalization", ["c", "s", "b", "m", "v"], ["bn"]).Node("Add", ["bn", "c"], ["y"]).Build();
        Assert.Equal(["Conv", "Mul", "Add", "Add"], Ops(def));
    }

    [Theory]
    [InlineData("SAME_UPPER", 8, 3, 2, new long[] { 0, 0, 1, 1 }, 4)]
    [InlineData("SAME_LOWER", 8, 3, 2, new long[] { 1, 1, 0, 0 }, 4)]
    [InlineData("VALID", 8, 3, 1, new long[] { 0, 0, 0, 0 }, 6)]
    public void Conv_AutoPad_BecomesExplicitPads(string mode, int size, int k, int stride, long[] pads, int outSize)
    {
        var def = new TestModel().Input("x", "N", 1, size, size).Output("y", "N", 2, outSize, outSize)
            .InitF32("w", [2, 1, k, k], Rnd(2 * k * k))
            .Node("Conv", ["x", "w"], ["y"], TestModel.AttrS("auto_pad", mode), TestModel.AttrInts("strides", stride, stride)).Build();
        var conv = Assert.Single(def.Graph("main").Nodes);
        Assert.Equal(pads, conv.Attributes.Get<long[]>("pads"));
        Assert.Equal(outSize, conv.Output.Shape[2].Fixed);
    }

    [Fact]
    public void Gemm_ConstantB_UsesZeroCopyWeightOrDerivedTranspose()
    {
        var def = new TestModel().Input("x", "N", 4).Output("y1", "N", 3).Output("y2", "N", 3)
            .InitF32("bt", [3, 4], Rnd(12)).InitF32("b", [4, 3], Rnd(12, 2)).InitF32("c", [3], [1, 2, 3])
            .Node("Gemm", ["x", "bt", "c"], ["y1"], TestModel.Attr("transB", 1)).Node("Gemm", ["x", "b"], ["y2"]).Build();
        var g = def.Graph("main");
        var mms = g.Nodes.Where(n => n.Op.Name == "MatMul").ToList();
        Assert.Equal(2, mms.Count);
        Assert.Equal("[3, 4]", mms[0].Inputs[1].Shape.ToString());
        Assert.Equal("[3, 4]", mms[1].Inputs[1].Shape.ToString());     // [K,N] transposed to the [N,K] layout
        Assert.Contains(g.Nodes, n => n.Op.Name == "Add");             // bias
        Assert.Equal(ValueKind.Weight, mms[1].Inputs[1].Kind);
    }

    [Fact]
    public void MatMul_Activations_UseBatchMatMul()
    {
        var def = new TestModel().Input("a", "N", 2, 3, 4).Input("b", "N", 2, 4, 5).Output("y", "N", 2, 3, 5).Node("MatMul", ["a", "b"], ["y"]).Build();
        Assert.Equal(["BatchMatMul"], Ops(def));
    }

    [Fact]
    public void Softmax_NonLastAxis_IsWrappedInPermutes()
    {
        var def = new TestModel().Opset(13).Input("x", "N", 3, 4).Output("y", "N", 3, 4).Node("Softmax", ["x"], ["y"], TestModel.Attr("axis", 1)).Build();
        Assert.Equal(["Permute", "Softmax", "Permute"], Ops(def));
    }

    [Fact]
    public void GlobalAveragePool_IsReduceMeanOverSpatialAxes()
    {
        var def = new TestModel().Input("x", "N", 3, 5, 5).Output("y", "N", 3, 1, 1).Node("GlobalAveragePool", ["x"], ["y"]).Build();
        var r = Assert.Single(def.Graph("main").Nodes);
        Assert.Equal("Reduce", r.Op.Name);
        Assert.Equal("mean", r.Attributes.Get<string>("mode"));
        Assert.Equal([2L, 3], r.Attributes.Get<long[]>("axes"));
    }

    [Fact]
    public void Placement_Groups_AreAssigned_AndRotate()
    {
        var m = new TestModel().Input("x", "N", 4).Output("y", "N", 4);
        string prev = "x";
        for (int i = 0; i < 6; i++)
        {
            m.InitF32($"w{i}", [4, 4], Rnd(16, i));
            m.Node("Gemm", [prev, $"w{i}"], [$"t{i}"], TestModel.Attr("transB", 1));
            prev = $"t{i}";
        }
        m.Node("Relu", [prev], ["y"]);
        var def = m.Build(new OnnxImportOptions { GroupSize = 2 });
        var groups = def.Graph("main").Nodes.Select(n => n.Group).Distinct().ToList();
        Assert.True(groups.Count >= 3, string.Join(",", groups));
        Assert.All(groups, g => Assert.StartsWith("onnx.", g));
    }

    [Fact]
    public void ConstantGraphOutput_IsRejected()
    {
        var m = new TestModel().Input("x", "N", 4).Output("y", "N", 4).Output("k", 1).InitF32("c", [1], [1f]).Node("Relu", ["x"], ["y"]).Node("Identity", ["c"], ["k"]);
        Assert.Throws<NotSupportedException>(() => m.Build());
    }

    [Fact]
    public void ReshapeThatMergesBatch_IsRejected()
    {
        var m = new TestModel().Input("x", "N", 3, 4).Output("y", 12).InitI64("s", [1], -1).Node("Reshape", ["x", "s"], ["y"]);
        var e = Assert.Throws<NotSupportedException>(() => m.Build());
        Assert.Contains("batch", e.Message);
    }

    [Fact]
    public void IfWithConstantCondition_InlinesTakenBranch()
    {
        ProtoWriter Branch(string op)
        {
            var g = new ProtoWriter().Msg(1, new ProtoWriter().Str(1, "x").Str(2, "o").Str(3, "n").Str(4, op)).Str(2, "br").Msg(12, TestModel.ValueInfo("o", 1, "N", 4));
            return g;
        }
        var cond = TestModel.TensorProto("cond", 9, [], [1]);
        var ifAttr = new ProtoWriter().Str(1, "then_branch").Msg(6, Branch("Relu")).Int(20, 5);
        var elseAttr = new ProtoWriter().Str(1, "else_branch").Msg(6, Branch("Neg")).Int(20, 5);
        var def = new TestModel().Input("x", "N", 4).Output("y", "N", 4).InitRaw(cond).Node("If", ["cond"], ["y"], ifAttr, elseAttr).Build();
        Assert.Equal(["Relu"], Ops(def));
    }
}
