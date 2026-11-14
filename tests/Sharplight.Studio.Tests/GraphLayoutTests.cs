using System.Diagnostics;
using Sharplight.Package;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.Tests;

/// <summary>The layered layout of the graph view.</summary>
public sealed class GraphLayoutTests
{
    [Fact]
    public void SmallGraph_PlacesProducersAboveConsumersWithoutOverlap()
    {
        using var tmp = new TempDir();
        var path = tmp.File("small.slm");
        TestPackages.WriteSmall(path);
        var graph = ModelPackageReader.Open(path).Definition.Graphs.Single();
        var layout = GraphLayout.Build(graph.Nodes, graph.Inputs, graph.Outputs, graph.Ir);

        Assert.Equal(graph.Inputs.Count + graph.NodeCount + graph.Outputs.Count, layout.Nodes.Count);
        var input = Assert.Single(layout.Nodes, n => n.Kind == LayoutNodeKind.Input);
        Assert.Equal(0, input.Rank);
        var output = Assert.Single(layout.Nodes, n => n.Kind == LayoutNodeKind.Output);
        Assert.Equal("y", output.Title);
        Assert.Equal(layout.RankCount - 1, output.Rank);
        foreach (var edge in layout.Edges)
        {
            Assert.True(layout.Nodes[edge.From].Rank < layout.Nodes[edge.To].Rank, $"{edge}");
            Assert.InRange(edge.Port, 0.0, 1.0);
        }

        foreach (var row in layout.Nodes.GroupBy(n => n.Rank))
        {
            var boxes = row.OrderBy(n => n.X).ToList();
            for (int i = 1; i < boxes.Count; i++)
            {
                Assert.True(boxes[i].X >= boxes[i - 1].X + boxes[i - 1].Width, "boxes of a row overlap");
            }
        }

        var silu = graph.Nodes.Single(n => n.Op == "Silu");
        var box = layout.NodeOf(silu.Id)!;
        Assert.Equal(OpCategory.Elementwise, box.Category);
        Assert.Equal("layers.1", box.Group);
        var matMul = layout.NodeOf(graph.Nodes.First(n => n.Op == "MatMul").Id)!;
        Assert.Equal(1, matMul.WeightCount);
        Assert.Equal(OpCategory.Linear, matMul.Category);
        Assert.Null(layout.NodeOf(12345));
    }

    [Fact]
    public void LargeChain_LaysOutQuickly()
    {
        const int Nodes = 100_000;
        using var tmp = new TempDir();
        var path = tmp.File("large.slm");
        TestPackages.WriteLarge(path, Nodes);
        var graph = ModelPackageReader.Open(path).Definition.Graphs.Single();
        var nodes = graph.Nodes;
        var ir = graph.Ir;
        var watch = Stopwatch.StartNew();
        var layout = GraphLayout.Build(nodes, graph.Inputs, graph.Outputs, ir);
        TestContext.Current.TestOutputHelper?.WriteLine($"{Nodes:N0} nodes laid out in {watch.ElapsedMilliseconds} ms, {layout.RankCount:N0} rows");
        Assert.Equal(Nodes + 2, layout.Nodes.Count);
        Assert.Equal(Nodes + 2, layout.RankCount);
        Assert.InRange(watch.ElapsedMilliseconds, 0, 5000);
    }

    [Fact]
    public void Attributes_SplitAtTopLevelCommas()
    {
        var pairs = AttributeText.Split("{axes=[0, 1], name=x, graph=<graph g>, scale=0.5}");
        Assert.Equal(["axes", "name", "graph", "scale"], pairs.Select(p => p.Key));
        Assert.Equal("[0, 1]", pairs[0].Value);
        Assert.Empty(AttributeText.Split("{}"));
    }

    [Fact]
    public void OpCategories_CoverCommonOps()
    {
        Assert.Equal(OpCategory.Linear, OpCategories.Of("MatMul"));
        Assert.Equal(OpCategory.Attention, OpCategories.Of("CachedAttention"));
        Assert.Equal(OpCategory.State, OpCategories.Of("CacheWrite"));
        Assert.Equal(OpCategory.Normalization, OpCategories.Of("RmsNorm"));
        Assert.Equal(OpCategory.Layout, OpCategories.Of("Reshape"));
        Assert.Equal(OpCategory.Elementwise, OpCategories.Of("Add"));
        Assert.Equal(OpCategory.Other, OpCategories.Of("If"));
    }
}
