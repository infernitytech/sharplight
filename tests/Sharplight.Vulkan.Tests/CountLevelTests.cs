using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Independent data-dependent counts computed on the GPU share one segment (count-level order), and boundary values between the
/// segments are copied on the device instead of through host memory.
/// </summary>
public sealed class CountLevelTests
{
    private const int Masks = 4;

    /// <summary>
    /// Four NonZero masks over one activation, each gathering rows of that activation by its count, and a second-level NonZero over
    /// the first gather: three count levels whichever the graph order.
    /// </summary>
    private static ComputeGraph Build(out Value x)
    {
        var tg = new TG();
        x = tg.InF("x", TG.T, 30);
        var h = tg.B.Binary("Mul", x, x);
        Value? first = null;
        for (int k = 0; k < Masks; k++)
        {
            var mask = tg.B.Binary("Greater", h, tg.Const($"threshold{k}", [0.2f + 0.15f * k], 1));
            var selected = NonZero(tg, mask);
            var rows = tg.B.Emit1("Reshape", [tg.B.Slice(selected, 1, 0, 1)], Attributes.From(("shape", new long[] { 0 })));
            var gathered = tg.B.Emit1("GatherAxis", [h, rows], Attributes.From(("axis", 0L)));
            tg.B.Output($"rows{k}", gathered);
            first ??= gathered;
        }

        var again = NonZero(tg, tg.B.Binary("Greater", first!, tg.Const("threshold", [0.5f], 1)));
        tg.B.Output("again", again);
        var againRows = tg.B.Emit1("Reshape", [tg.B.Slice(again, 1, 0, 1)], Attributes.From(("shape", new long[] { 0 })));
        tg.B.Output("picked", tg.B.Emit1("GatherAxis", [first!, againRows], Attributes.From(("axis", 0L))));
        return tg.B.Build();
    }

    private static Value NonZero(TG tg, Value x)
    {
        string count = tg.B.DynamicSymbol("nonzero", x.Shape.Dims.Aggregate((Dim)1, (a, b) => a * b));
        return tg.B.Emit1("NonZero", [x], Attributes.From(("count_symbol", count)));
    }

    /// <summary>The grouped and forwarded execution against the CPU oracle.</summary>
    [Theory]
    [InlineData(40)]
    [InlineData(64)]
    public void MatchesCpu(int rows)
    {
        var g = Build(out var x);
        Diff.Check("NonZero levels", g, new Step(rows, 0, (x, Rnd.F(rows * 30L, 11))), atol: 1e-6, rtol: 1e-6, maxTokens: 64, what: $"rows={rows}");
    }

    /// <summary>One segment per count level, whichever order the graph lists the nodes in (never more than in graph order).</summary>
    [Fact]
    public void SegmentPerLevel()
    {
        Gpu.Require();
        var g = Build(out _);
        Assert.Equal(3, Segments(g, group: true));
        Assert.True(Segments(g, group: false) >= 3);
    }

    private static int Segments(ComputeGraph g, bool group)
    {
        using var model = CompiledModel.Compile(g, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = Gpu.Vulkan! }],
            Compile = new CompileOptions { MaxTokens = 64 },
            ReservePerDevice = 0,
            GroupDataDependentCounts = group,
        });
        return model.Plan.Segments.Count;
    }
}
