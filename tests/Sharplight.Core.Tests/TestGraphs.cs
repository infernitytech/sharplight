using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Core.Tests;

/// <summary>Minimal graphs for tests that only need graphs to exist.</summary>
internal static class TestGraphs
{
    /// <summary>A graph named <paramref name="name"/> that passes its input through a negation.</summary>
    public static ComputeGraph Trivial(string name)
    {
        var builder = new GraphBuilder(name, _ => null);
        var x = builder.Input("x", InputRole.Other, DataType.F32, new Shape(4));
        builder.Output("y", builder.Emit1("Neg", [x]));
        return builder.Build();
    }
}
