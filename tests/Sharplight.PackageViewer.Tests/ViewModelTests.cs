using Sharplight.Graph;
using Sharplight.Package;
using Sharplight.PackageViewer.ViewModels;
using Xunit;

namespace Sharplight.PackageViewer.Tests;

/// <summary>The view models populate from a package written by <see cref="ModelPackage.Save"/>.</summary>
public sealed class ViewModelTests
{
    [Fact]
    public Task SmallPackagePopulatesEveryTab() => HeadlessApp.RunAsync(async () =>
    {
        using var tmp = new TempDir();
        var path = tmp.File("small.slm");
        TestPackages.WriteSmall(path);

        var main = new MainWindowViewModel();
        Assert.True(await main.OpenAsync(path), main.Status);
        var package = main.Package!;
        Assert.Equal($"small.slm - {MainWindowViewModel.AppTitle}", main.Title);

        var overview = package.Overview;
        Assert.Equal(2, overview.FormatVersion);
        Assert.Null(overview.DefinitionError);
        Assert.Equal(["ARTS", "ARTS", "DEFN", "WDAT", "WIDX"], overview.Sections.Select(s => s.Kind).Order(StringComparer.Ordinal));
        Assert.Equal(3, overview.ValueCount);
        overview.Filter = "tokenizer";
        var tokens = Assert.Single(overview.Values);
        Assert.Equal("metadata", tokens.Source);
        Assert.Contains("a, b, c", tokens.Detail, StringComparison.Ordinal);

        var graph = Assert.Single(package.Graphs.Graphs);
        Assert.Same(graph, package.Graphs.SelectedGraph);
        await graph.LoadAsync();
        Assert.Equal(graph.NodeCount, graph.Nodes.Count);
        Assert.Contains(graph.OpHistogram, o => o.Op == "MatMul" && o.Count == 2);
        graph.SelectedOp = graph.OpHistogram.Single(o => o.Op == "Silu");
        await graph.Filtering;
        Assert.Equal("Silu", Assert.Single(graph.Nodes).Op);

        var weights = package.Weights;
        Assert.Equal(2, weights.Weights.Count);
        Assert.Equal(["f32", "f16"], weights.TypeTotals.Select(t => t.Type));
        weights.Filter = "w1";
        Assert.Equal("w1", Assert.Single(weights.Weights).Name);

        var artifacts = package.Artifacts;
        var vulkan = Assert.Single(artifacts.Providers);
        Assert.Equal("vulkan", vulkan.Label);
        Assert.Equal(["binary", "code"], vulkan.Children.Select(t => t.Label));
        var codeSet = vulkan.Children[1].Children.Single();
        artifacts.SelectedNode = codeSet;
        var set = artifacts.SelectedSet!;
        Assert.Equal(2, set.EntryCount);
        Assert.Contains(set.Descriptor, d => d.Key == "target" && d.Value == "spirv_1_6");
        set.EntryFilter = "0001";
        Assert.Equal(TestPackages.CodeKernelKey, Assert.Single(set.Entries).Key);
        artifacts.SelectedNode = vulkan;
        Assert.Null(artifacts.SelectedSet);
    });

    [Fact]
    public Task IrViewNavigatesValuesAndFollowsTheNodeTable() => HeadlessApp.RunAsync(async () =>
    {
        using var tmp = new TempDir();
        var path = tmp.File("small.slm");
        TestPackages.WriteSmall(path);
        var main = new MainWindowViewModel();
        Assert.True(await main.OpenAsync(path), main.Status);
        var graph = main.Package!.Graphs.SelectedGraph!;
        await graph.LoadAsync();
        var ir = graph.Ir!;

        Assert.Equal(GraphDumpLineKind.Graph, ir.Lines[0].Line.Kind);
        Assert.Contains(ir.Lines, l => l.Line.Kind == GraphDumpLineKind.Weight && l.Text.Contains("w0", StringComparison.Ordinal));
        Assert.Contains(ir.Lines, l => l.Line.Kind == GraphDumpLineKind.State && l.Text.Contains("KeyCache", StringComparison.Ordinal));

        // Node table -> IR line.
        var silu = graph.Nodes.Single(n => n.Op == "Silu");
        graph.SelectedNode = silu;
        Assert.Equal(silu.Id, ir.SelectedLine!.Line.NodeId);
        Assert.Contains("Silu(", ir.SelectedLine.Text, StringComparison.Ordinal);

        // Go to the producer of the Silu input (the second MatMul), then IR line -> node table.
        var producer = Assert.Single(ir.Producers);
        ir.GoTo(producer.Target);
        Assert.Equal("MatMul", graph.SelectedNode!.Op);
        Assert.Contains(ir.Consumers, c => c.Target.Line.NodeId == silu.Id);
        Assert.Contains(ir.Neighborhood, n => n.Target.Line.Kind == GraphDumpLineKind.Weight);

        ir.BackCommand.Execute(null);
        Assert.Same(silu, graph.SelectedNode);

        ir.SearchText = "output y";
        ir.FindNextCommand.Execute(null);
        Assert.Equal(GraphDumpLineKind.Output, ir.SelectedLine!.Line.Kind);
        ir.SearchText = "no such text";
        ir.FindNextCommand.Execute(null);
        Assert.Equal("not found", ir.SearchStatus);
    });

    [Fact]
    public Task MissingFileReportsAnError() => HeadlessApp.RunAsync(async () =>
    {
        var main = new MainWindowViewModel();
        Assert.False(await main.OpenAsync(Path.Combine(Path.GetTempPath(), "does-not-exist.slm")));
        Assert.Null(main.Package);
        Assert.StartsWith("Cannot open", main.Status, StringComparison.Ordinal);
    });
}
