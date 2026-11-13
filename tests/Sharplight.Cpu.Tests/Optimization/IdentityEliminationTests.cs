using Sharplight.Graph;
using Sharplight.Graph.Optimization;
using Sharplight.Models;
using Sharplight.Tensors;
using static Sharplight.Cpu.Tests.Optimization.OptimizationKit;

namespace Sharplight.Cpu.Tests.Optimization;

public class IdentityEliminationTests
{
    [Fact]
    public void BypassesNoOpViewsCastsAndCopies()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, 4, 6);
        var w = tg.WF("w", Kit.Rand(24, 1), 4, 6);
        var v = tg.B.Reshape(x, 0, 4, 6);
        v = tg.B.Permute(v, 0, 1, 2);
        v = tg.B.Slice(v, 2, 0, 6);
        v = tg.B.Emit1("Cast", [v], Attributes.From(("type", "f32")));
        v = tg.B.Emit1("Expand", [v], Attributes.From(("shape", new long[] { 1, 4, 6 })));
        v = tg.B.Emit1("Tile", [v], Attributes.From(("repeats", new long[] { 1, 1, 1 })));
        v = tg.B.Emit1("Pad", [v], Attributes.From(("pads", new long[6]), ("mode", "constant"), ("value", 0f)));
        var product = tg.B.MatMul(v, w);
        tg.B.Output("y", tg.B.Unary("Neg", tg.B.Contiguous(product)));
        var g = tg.B.Build();

        var optimized = GraphOptimizer.Default.Optimize(g).Graph;

        Assert.Equal(["MatMul", "Neg"], optimized.Nodes.Select(n => n.Op.Name));
        Assert.Same(x, optimized.Nodes[0].Inputs[0]);
        AssertSameResults(g, optimized, new Dictionary<Value, byte[]> { [x] = Bytes(Kit.Rand(48, 2)) }, Kit.P(2));
    }

    [Fact]
    public void KeepsCopiesOfViewsAndStatesAndGraphOutputs()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", 3, 4);
        var state = tg.State("s", DataType.F32, StateKind.Other, 3, 4);
        var copyOfView = tg.B.Contiguous(tg.B.Permute(x, 1, 0));          // materializes a strided view: kept
        var stateView = tg.B.Reshape(state, 3, 4);                         // a state never replaces a computed value
        var lossyCast = tg.B.Emit1("Cast", [tg.B.Emit1("Cast", [x], Attributes.From(("type", "i64")))], Attributes.From(("type", "i64")));
        tg.B.Output("a", tg.B.Unary("Neg", copyOfView));
        tg.B.Output("b", tg.B.Add(stateView, x));
        tg.B.Output("c", tg.B.Emit1("Cast", [lossyCast], Attributes.From(("type", "f32"))));
        tg.B.Output("d", tg.B.Reshape(x, 3, 4));                           // identity, but a graph output
        var g = tg.B.Build();

        var optimized = new IdentityElimination().Run(g);

        Assert.Same(g, optimized);
    }
}
