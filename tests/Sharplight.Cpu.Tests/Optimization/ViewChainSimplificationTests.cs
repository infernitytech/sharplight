using Sharplight.Graph;
using Sharplight.Graph.Optimization;
using static Sharplight.Cpu.Tests.Optimization.OptimizationKit;

namespace Sharplight.Cpu.Tests.Optimization;

public class ViewChainSimplificationTests
{
    [Fact]
    public void CollapsesReshapeChainsWithASymbolicOuterDimension()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, 4, 8);
        var flat = tg.B.Reshape(x, 0, 32);
        var split = tg.B.Reshape(flat, 0, 2, 16);
        tg.B.Output("y", tg.B.Unary("Neg", split));
        tg.B.Output("flat", tg.B.Unary("Abs", flat));     // the inner reshape stays for its other use
        var g = tg.B.Build();

        var optimized = GraphOptimizer.Default.Optimize(g).Graph;

        var reshapes = optimized.Nodes.Where(n => n.Op.Name == "Reshape").ToList();
        Assert.Equal(2, reshapes.Count);
        Assert.All(reshapes, r => Assert.Same(x, r.Inputs[0]));
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Kit.Rand(96, 1)) }, Kit.P(3));
    }

    [Fact]
    public void CollapsesReshapesThatChangeTheOuterSymbol()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, 4, 8);
        var rows = tg.B.Emit1("Reshape", [x], Attributes.From(("shape", new long[] { 0, 8 }), ("outer_symbol", "4*T")));
        var back = tg.B.Emit1("Reshape", [rows], Attributes.From(("shape", new long[] { 0, 2, 16 }), ("outer_symbol", "T")));
        tg.B.Output("y", tg.B.Unary("Neg", back));
        var g = tg.B.Build();

        var optimized = GraphOptimizer.Default.Optimize(g).Graph;

        Assert.Equal(1, Count(optimized, "Reshape"));
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Kit.Rand(64, 2)) }, Kit.P(2));
    }

    [Fact]
    public void ComposesPermutationsAndDropsTheIdentity()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", 2, 3, 4);
        var once = tg.B.Permute(x, 2, 0, 1);
        var twice = tg.B.Permute(once, 1, 0, 2);
        var back = tg.B.Permute(tg.B.Permute(x, 1, 2, 0), 2, 0, 1);   // identity
        tg.B.Output("y", tg.B.Contiguous(twice));
        tg.B.Output("z", tg.B.Unary("Neg", back));
        var g = tg.B.Build();

        var optimized = GraphOptimizer.Default.Optimize(g).Graph;

        Assert.Equal(1, Count(optimized, "Permute"));
        Assert.Same(x, optimized.Outputs["z"].Producer!.Inputs[0]);
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Kit.Rand(24, 3)) });
    }

    [Fact]
    public void CollapsesSlicesAlongTheSameAxis()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", 3, 16);
        var inner = tg.B.Slice(x, -1, 4, 10);
        var outer = tg.B.Slice(inner, 1, 3, 5);
        var otherAxis = tg.B.Slice(tg.B.Slice(x, 1, 2, 8), 0, 1, 2);   // different axes: stays a chain
        tg.B.Output("y", tg.B.Contiguous(outer));
        tg.B.Output("z", tg.B.Contiguous(otherAxis));
        var g = tg.B.Build();

        var optimized = GraphOptimizer.Default.Optimize(g).Graph;

        Assert.Equal(3, Count(optimized, "Slice"));
        var y = optimized.Outputs["y"].Producer!.Inputs[0].Producer!;
        Assert.Same(x, y.Inputs[0]);
        Assert.Equal(7L, y.Attributes.Get<long>("start"));
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Kit.Rand(48, 4)) });
    }
}
