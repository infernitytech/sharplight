using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.VisualTree;
using Sharplight.PackageViewer.ViewModels;
using Sharplight.PackageViewer.Views;
using Xunit;

namespace Sharplight.PackageViewer.Tests;

/// <summary>The real window renders every tab headlessly; a 100k-node graph stays virtualized and fast.</summary>
public sealed class MainWindowTests
{
    private const int LargeNodeCount = 100_000;

    [Fact]
    public Task EveryTabRenders() => HeadlessApp.RunAsync(async () =>
    {
        using var tmp = new TempDir();
        var path = tmp.File("small.slm");
        TestPackages.WriteSmall(path);
        var main = new MainWindowViewModel();
        var window = new MainWindow { DataContext = main };
        window.Show();
        Assert.True(await main.OpenAsync(path), main.Status);
        await main.Package!.Graphs.SelectedGraph!.LoadAsync();
        Assert.Equal(main.Title, window.Title);
        var artifacts = main.Package.Artifacts;
        artifacts.SelectedNode = artifacts.Providers[0].Children[0].Children[0];

        var tabs = window.GetVisualDescendants().OfType<TabControl>().Single();
        for (int i = 0; i < tabs.ItemCount; i++)
        {
            tabs.SelectedIndex = i;
            HeadlessApp.Render(window);
            Assert.True(window.GetVisualDescendants().OfType<DataGrid>().Any(g => g.IsEffectivelyVisible), $"tab {i} shows no table");
        }

        tabs.SelectedIndex = 1;
        HeadlessApp.Render(window);
        var irList = IrList(window);
        Assert.Equal(main.Package.Graphs.SelectedGraph.Ir!.Lines.Count, irList.ItemCount);
        window.Close();
    });

    [Fact]
    public Task LargeGraphIsVirtualizedAndResponsive()
    {
        var output = TestContext.Current.TestOutputHelper;
        return HeadlessApp.RunAsync(async () => await RunLargeGraph(output));
    }

    private static async Task RunLargeGraph(ITestOutputHelper? output)
    {
        using var tmp = new TempDir();
        var path = tmp.File("large.slm");
        TestPackages.WriteLarge(path, LargeNodeCount);
        var main = new MainWindowViewModel();
        var window = new MainWindow { DataContext = main };
        window.Show();

        var timings = new List<string>();
        var watch = Stopwatch.StartNew();
        Assert.True(await main.OpenAsync(path), main.Status);
        timings.Add($"open {watch.ElapsedMilliseconds} ms");
        var graph = main.Package!.Graphs.SelectedGraph!;

        watch.Restart();
        await graph.LoadAsync();
        timings.Add($"node table + IR {watch.ElapsedMilliseconds} ms");
        Assert.Equal(LargeNodeCount, graph.Nodes.Count);

        var tabs = window.GetVisualDescendants().OfType<TabControl>().Single();
        watch.Restart();
        tabs.SelectedIndex = 1;
        HeadlessApp.Render(window);
        timings.Add($"render graphs tab {watch.ElapsedMilliseconds} ms");
        var nodeGrid = window.GetVisualDescendants().OfType<DataGrid>().Single(g => ReferenceEquals(g.ItemsSource, graph.Nodes));
        int realizedRows = nodeGrid.GetVisualDescendants().OfType<DataGridRow>().Count();
        Assert.InRange(realizedRows, 1, 500);
        int realizedLines = IrList(window).GetRealizedContainers().Count();
        Assert.InRange(realizedLines, 1, 500);

        watch.Restart();
        graph.NodeFilter = "op:Silu layers.42";
        await graph.Filtering;
        HeadlessApp.Render(window);
        timings.Add($"filter {watch.ElapsedMilliseconds} ms");
        Assert.Equal(LargeNodeCount / 100 / 2, graph.Nodes.Count);

        watch.Restart();
        graph.SelectedNode = graph.Nodes[^1];
        HeadlessApp.Render(window);
        timings.Add($"select node + jump to IR {watch.ElapsedMilliseconds} ms");
        Assert.Equal(graph.SelectedNode.Id, graph.Ir!.SelectedLine!.Line.NodeId);

        watch.Restart();
        graph.Ir.SearchText = "output y";
        graph.Ir.FindNextCommand.Execute(null);
        HeadlessApp.Render(window);
        timings.Add($"find {watch.ElapsedMilliseconds} ms");

        output?.WriteLine($"{LargeNodeCount:N0} nodes: {string.Join(", ", timings)}; "
            + $"realized {realizedRows} grid rows, {realizedLines} IR lines");
        window.Close();
    }

    private static ListBox IrList(MainWindow window) =>
        window.GetVisualDescendants().OfType<IrView>().Single().GetVisualDescendants().OfType<ListBox>().First();
}
