using Sharplight.Graph;
using Sharplight.Graph.Optimization;
using Sharplight.Tensors;
using static Sharplight.Cpu.Tests.Optimization.OptimizationKit;

namespace Sharplight.Cpu.Tests.Optimization;

public class ConstantFoldingTests
{
    private static readonly GraphOptimizer s_foldOnly = new("test", [new ConstantFolding(), new DeadNodeElimination()]);

    private static float[] Values(int n, int seed)
    {
        // Includes the awkward cases: signed zeros, halves (rounding), negative values and an infinity.
        var v = Kit.Rand(n, seed, 4f);
        v[0] = -0f;
        v[1] = 0f;
        v[2] = 2.5f;
        v[3] = -1.5f;
        v[4] = float.PositiveInfinity;
        return v;
    }

    public static TheoryData<string> UnaryOps => ["Neg", "Abs", "Relu", "Square", "Recip", "Sqrt", "Rsqrt", "Floor", "Ceil", "Round", "Sign", "Not", "IsNaN", "IsInf"];

    public static TheoryData<string> BinaryOps =>
        ["Add", "Sub", "Mul", "Div", "Max", "Min", "Mod", "PRelu", "Equal", "NotEqual", "Less", "LessEqual", "Greater", "GreaterEqual", "And", "Or", "Xor"];

    [Theory]
    [MemberData(nameof(UnaryOps))]
    public void FoldsExactUnaryOpsBitExactly(string op)
    {
        var tg = new TestGraph();
        var x = tg.InF("x", 6, 8);
        var c = ConstF(tg.B, "c", Values(48, 1), 6, 8);
        tg.B.Output("y", tg.B.Mul(x, tg.B.Unary(op, c)));   // x = 1 keeps signed zeros and NaNs visible
        var g = tg.B.Build();

        var optimized = s_foldOnly.Optimize(g).Graph;

        Assert.Equal(0, Count(optimized, op));
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Enumerable.Repeat(1f, 48).ToArray()) });
    }

    [Theory]
    [MemberData(nameof(BinaryOps))]
    public void FoldsBroadcastingBinaryOpsBitExactly(string op)
    {
        var tg = new TestGraph();
        var x = tg.InF("x", 6, 8);
        var a = ConstF(tg.B, "a", Values(48, 2), 6, 8);
        var bValues = Values(8, 3);
        bValues[0] = 0f;                               // pairs with -0 in a
        var b = ConstF(tg.B, "b", bValues, 8);
        tg.B.Output("y", tg.B.Mul(x, tg.B.Binary(op, a, b)));
        var g = tg.B.Build();

        var optimized = s_foldOnly.Optimize(g).Graph;

        bool signedZeroPair = op is "Max" or "Min";   // -0 vs +0 operands: the CPU kernels disagree, so it must not fold
        int outerMul = op == "Mul" ? 1 : 0;
        Assert.Equal((signedZeroPair ? 1 : 0) + outerMul, Count(optimized, op));
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Enumerable.Repeat(1f, 48).ToArray()) });
    }

    [Fact]
    public void FoldsDataMovementChainsOverIntegers()
    {
        var tg = new TestGraph();
        var x = tg.In("x", DataType.I32, 3, 4);
        var c = ConstI(tg.B, "c", [.. Enumerable.Range(0, 12)], 4, 3);
        var permuted = tg.B.Contiguous(tg.B.Permute(c, 1, 0));                                   // [3, 4]
        var sliced = tg.B.Slice(permuted, 1, 1, 2);                                               // [3, 2]
        var joined = tg.B.Concat(1, sliced, sliced);                                             // [3, 4]
        var tiled = tg.B.Emit1("Tile", [tg.B.Reshape(joined, 1, 12)], Attributes.From(("repeats", new long[] { 1, 1 })));
        var expanded = tg.B.Emit1("Expand", [tiled], Attributes.From(("shape", new long[] { 1, 12 })));
        var indices = ConstI(tg.B, "i", [2, -1, 0], 3);
        var gathered = tg.B.Emit1("GatherAxis", [tg.B.Reshape(expanded, 3, 4), indices], Attributes.From(("axis", 0L)));
        var casted = tg.B.Emit1("Cast", [gathered], Attributes.From(("type", "f32")));
        tg.B.Output("y", tg.B.Add(tg.B.Emit1("Cast", [x], Attributes.From(("type", "f32"))), casted));
        var g = tg.B.Build();

        var optimized = s_foldOnly.Optimize(g).Graph;

        Assert.Equal(["Cast", "Add"], optimized.Nodes.Select(n => n.Op.Name));
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Enumerable.Range(0, 12).ToArray()) });
    }

    [Fact]
    public void FoldsCastsLikeTheCpuBackend()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", 8);
        var c = ConstF(tg.B, "c", [1.5f, -2.7f, 3e10f, -0.4f, 255.9f, 300f, float.NaN, -129f], 8);
        Value Through(string type) => tg.B.Emit1("Cast", [tg.B.Emit1("Cast", [c], Attributes.From(("type", type)))], Attributes.From(("type", "f32")));
        var sum = tg.B.Add(tg.B.Add(Through("i32"), Through("i64")), tg.B.Add(Through("u8"), Through("i8")));
        tg.B.Output("y", tg.B.Mul(x, tg.B.Add(sum, Through("bool"))));
        var g = tg.B.Build();

        var optimized = s_foldOnly.Optimize(g).Graph;

        Assert.Equal(0, Count(optimized, "Cast"));
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Enumerable.Repeat(1f, 8).ToArray()) });
    }

    [Fact]
    public void FoldsWhereAndStaticFill()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", 4, 3);
        var range = tg.B.Emit1("Fill", [], Attributes.From(("shape", new long[] { 4, 3 }), ("start", 1.0), ("delta", 2.0), ("type", "f32")));
        var mask = ConstF(tg.B, "mask", [1, 0, 1], 3);
        var other = ConstF(tg.B, "other", [-1], 1);
        tg.B.Output("y", tg.B.Add(x, tg.B.Emit1("Where", [mask, range, other])));
        var g = tg.B.Build();

        var optimized = s_foldOnly.Optimize(g).Graph;

        Assert.Equal(["Add"], optimized.Nodes.Select(n => n.Op.Name));
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Kit.Rand(12, 6)) });
    }

    [Fact]
    public void LeavesTranscendentalOpsSymbolicShapesOutputsAndLargeResultsAlone()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, 4);
        var c = ConstF(tg.B, "c", [1, 2, 3, 4], 4);
        var exp = tg.B.Unary("Exp", c);                                              // CPU uses a vectorized approximation
        var symbolic = tg.B.Emit1("Fill", [], Attributes.From(("shape", new long[] { 0, 4 }), ("outer_symbol", "T"), ("start", 1.0), ("delta", 0.0)));
        var large = tg.B.Emit1("Expand", [c], Attributes.From(("shape", new long[] { 1 << 15, 4 })));   // 512 KiB
        tg.B.Output("neg", tg.B.Unary("Neg", c));                                    // a graph output: never replaced by a constant
        tg.B.Output("y", tg.B.Add(tg.B.Add(x, exp), symbolic));
        tg.B.Output("large", tg.B.Emit1("ReduceSum", [tg.B.Emit1("Permute", [large], Attributes.From(("perm", new[] { 1, 0 })))]));
        var g = tg.B.Build();

        var optimized = s_foldOnly.Optimize(g).Graph;

        Assert.Equal(g.Nodes.Select(n => n.Op.Name), optimized.Nodes.Select(n => n.Op.Name));
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Kit.Rand(12, 7)) }, Kit.P(3));
    }

    [Fact]
    public void RespectsTheSizeCap()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", 64, 4);
        var c = ConstF(tg.B, "c", [1, 2, 3, 4], 4);
        tg.B.Output("y", tg.B.Add(x, tg.B.Emit1("Expand", [c], Attributes.From(("shape", new long[] { 64, 4 })))));
        var g = tg.B.Build();

        var capped = new GraphOptimizer("test", [new ConstantFolding { MaxOutputBytes = 512 }]).Optimize(g).Graph;
        var folded = new GraphOptimizer("test", [new ConstantFolding(), new DeadNodeElimination()]).Optimize(g).Graph;

        Assert.Equal(1, Count(capped, "Expand"));
        Assert.Equal(0, Count(folded, "Expand"));
    }
}
