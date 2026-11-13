using Sharplight.Graph;
using Sharplight.Graph.Optimization;
using Sharplight.Models;
using Sharplight.Tensors;
using static Sharplight.Cpu.Tests.Optimization.OptimizationKit;

namespace Sharplight.Cpu.Tests.Optimization;

public class DeadNodeEliminationTests
{
    [Fact]
    public void RemovesUnusedChainsAndKeepsOutputsInputsAndSideEffects()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, 4);
        var unused = tg.InF("unused", TestGraph.T, 4);
        var state = tg.State("s", DataType.F32, StateKind.Other, 8, 4);
        var dead = tg.B.Silu(tg.B.Unary("Exp", x));
        _ = tg.B.Mul(dead, x);
        var stateRead = tg.B.Reshape(state, 32);      // reads state: kept although unused
        _ = tg.B.Unary("Neg", stateRead);              // pure consumer of the state read: removable
        tg.B.CacheWrite(state, x);                      // side effect: kept
        var y = tg.B.Unary("Abs", x);
        tg.B.Output("y", y);
        var g = tg.B.Build();

        var optimized = new DeadNodeElimination().Run(g);

        Assert.Equal(["Reshape", "CacheWrite", "Abs"], optimized.Nodes.Select(n => n.Op.Name));
        Assert.Equal(g.Inputs, optimized.Inputs);
        Assert.Contains(unused, optimized.Inputs);
        Assert.Equal(y.Id, optimized.Outputs["y"].Id);
        Assert.NotSame(y, optimized.Outputs["y"]); // the source graph is left untouched
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Kit.Rand(8, 1)), [unused] = Bytes(Kit.Rand(8, 2)) },
            Kit.P(2, 0), Kit.P(2, 2));
    }

    [Fact]
    public void ReturnsTheSameGraphWhenNothingIsDead()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", 4);
        tg.B.Output("y", tg.B.Unary("Abs", x));
        var g = tg.B.Build();
        Assert.Same(g, new DeadNodeElimination().Run(g));
    }

    [Fact]
    public void KeepsNodesOwningDataDependentSymbols()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", 6);
        var count = tg.B.DynamicSymbol("nz", 6);
        _ = tg.B.Emit1("NonZero", [x], Attributes.From(("count_symbol", count)));
        tg.B.Output("y", tg.B.Unary("Abs", x));
        var g = tg.B.Build();
        Assert.Equal(1, Count(new DeadNodeElimination().Run(g), "NonZero"));
    }
}
