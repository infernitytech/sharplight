using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Core.Tests.Runtime;

/// <summary>
/// <see cref="PlacementPlanner.Estimate"/>: the activation bound of a decoder does not grow with the shared KV cache, whose few large
/// buffers need no scratch of their own (a share of a 92 GB default pool made the host shrink the pool too far, benchmarks.md §105).
/// </summary>
public sealed class FootprintEstimateTests
{
    [Fact]
    public void ActivationsDoNotGrowWithTheKvPool()
    {
        var graph = CachedAttention();
        var options = new CompileOptions { MaxTokens = 64, ContextLength = 256 };
        var small = PlacementPlanner.Estimate(graph, options, slots: 8, kvCacheRows: 1024);
        var large = PlacementPlanner.Estimate(graph, options, slots: 8, kvCacheRows: 1024 * 1024);

        Assert.Equal(1024L * 1024 / 1024 * small.StateBytes, large.StateBytes);
        Assert.Equal(small.ActivationBytes, large.ActivationBytes);
    }

    private static ComputeGraph CachedAttention()
    {
        var b = new GraphBuilder("attn", _ => null);
        var t = Dim.Sym("T");
        var q = b.Input("q", InputRole.Other, DataType.F32, new Shape(t, 4, 16));
        var k = b.Input("k", InputRole.Other, DataType.F32, new Shape(t, 2, 16));
        var v = b.Input("v", InputRole.Other, DataType.F32, new Shape(t, 2, 16));
        var kc = b.State(new StateDefinition { Name = "kc", Type = DataType.F16, Shape = new Shape(Dim.Sym("ctx"), 2, 16), Kind = StateKind.KeyCache });
        var vc = b.State(new StateDefinition { Name = "vc", Type = DataType.F16, Shape = new Shape(Dim.Sym("ctx"), 2, 16), Kind = StateKind.ValueCache });
        b.CacheWrite(kc, k);
        b.CacheWrite(vc, v);
        b.RuntimeScalar("n_past");
        b.Output("o", b.Emit1("CachedAttention", [q, kc, vc], Attributes.From([("scale", 0.25f), ("offset", "n_past")])));
        return b.Build();
    }
}
