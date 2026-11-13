using Sharplight.Graph;
using Sharplight.Graph.Optimization;
using Sharplight.Models;
using Sharplight.Tensors;
using static Sharplight.Cpu.Tests.Optimization.OptimizationKit;

namespace Sharplight.Cpu.Tests.Optimization;

public class CommonSubexpressionEliminationTests
{
    [Fact]
    public void MergesEqualNodesTransitively()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, 8);
        var a = tg.B.Silu(tg.B.Scale(x, 0.5f, 1f));
        var b = tg.B.Silu(tg.B.Scale(x, 0.5f, 1f));
        var c = tg.B.Scale(x, 0.5f, 2f);              // different attribute: stays
        tg.B.Output("y", tg.B.Add(tg.B.Mul(a, b), c));
        var g = tg.B.Build();

        var optimized = GraphOptimizer.Default.Optimize(g).Graph;

        Assert.Equal(2, Count(optimized, "Scale"));
        Assert.Equal(1, Count(optimized, "Silu"));
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Kit.Rand(24, 3)) }, Kit.P(3));
    }

    [Fact]
    public void AttributeValuesOfDifferentTypesAreDistinguished()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", 4);
        var a = tg.B.Emit1("Scale", [x], Attributes.From(("scale", 2f), ("bias", 0f)));
        var b = tg.B.Emit1("Scale", [x], Attributes.From(("scale", 2.0), ("bias", 0f)));
        tg.B.Output("y", tg.B.Add(a, b));
        var g = tg.B.Build();
        Assert.Equal(2, Count(new CommonSubexpressionElimination().Run(g), "Scale"));
    }

    [Fact]
    public void NeverMergesStateReadsAcrossWrites()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, 4);
        var state = tg.State("s", DataType.F32, StateKind.Other, 8, 4);
        var before = tg.B.Contiguous(tg.B.Reshape(state, 32));
        tg.B.CacheWrite(state, x);
        var after = tg.B.Contiguous(tg.B.Reshape(state, 32));
        tg.B.Output("before", before);
        tg.B.Output("after", after);
        var g = tg.B.Build();

        var optimized = GraphOptimizer.Default.Optimize(g).Graph;

        Assert.Equal(2, Count(optimized, "Reshape"));
        Assert.Equal(2, Count(optimized, "Contiguous"));
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Kit.Rand(8, 4)) }, Kit.P(2, 0), Kit.P(2, 2));
    }

    [Fact]
    public void KeepsDuplicatesThatAreGraphOutputs()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", 4);
        tg.B.Output("a", tg.B.Unary("Abs", x));
        tg.B.Output("b", tg.B.Unary("Abs", x));
        var g = tg.B.Build();
        var optimized = new CommonSubexpressionElimination().Run(g);
        Assert.Same(g, optimized);
    }

    [Fact]
    public void MergesEqualConstants()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", 4);
        var one = ConstF(tg.B, "one", [1, 2, 3, 4], 4);
        var same = ConstF(tg.B, "same", [1, 2, 3, 4], 4);
        var other = ConstF(tg.B, "other", [1, 2, 3, 5], 4);
        var y = tg.B.Add(tg.B.Mul(x, one), tg.B.Mul(x, same));
        tg.B.Output("y", tg.B.Add(y, tg.B.Mul(x, other)));
        var g = tg.B.Build();

        var optimized = new GraphOptimizer("test", [new CommonSubexpressionElimination(), new DeadNodeElimination()]).Optimize(g).Graph;

        Assert.Equal(2, Count(optimized, "Mul"));
        var constants = optimized.Nodes.SelectMany(n => n.Inputs).Where(v => v.Kind == ValueKind.Constant).Distinct().ToList();
        Assert.Equal(2, constants.Count);
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Kit.Rand(4, 5)) });
    }
}
