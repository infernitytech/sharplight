using System.Diagnostics;
using Avalonia.Controls;
using Avalonia.VisualTree;
using Sharplight.PackageViewer.ViewModels;
using Sharplight.PackageViewer.Views;
using Xunit;

namespace Sharplight.PackageViewer.Tests;

/// <summary>
/// Opens a real package named by <c>SHARPLIGHT_VIEWER_PACKAGE</c> (e.g. one written by <c>sharplight-convert convert</c> and
/// <c>precompile</c>), loads every graph and renders every tab, and reports the timings. Skipped when the variable is not set.
/// </summary>
public sealed class RealPackageTests
{
    private const string PackageVariable = "SHARPLIGHT_VIEWER_PACKAGE";

    [Fact]
    public Task RealPackageLoads()
    {
        var path = Environment.GetEnvironmentVariable(PackageVariable);
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            Assert.Skip($"{PackageVariable} does not name a package");
        }

        var output = TestContext.Current.TestOutputHelper;
        return HeadlessApp.RunAsync(async () => await Run(path, output));
    }

    private static async Task Run(string path, ITestOutputHelper? output)
    {
        var main = new MainWindowViewModel();
        var window = new MainWindow { DataContext = main };
        window.Show();
        var timings = new List<string>();
        var watch = Stopwatch.StartNew();
        Assert.True(await main.OpenAsync(path), main.Status);
        timings.Add($"open (header, index, definition, artifact directories) {watch.ElapsedMilliseconds} ms");
        var package = main.Package!;
        Assert.Null(package.Overview.DefinitionError);

        foreach (var graph in package.Graphs.Graphs)
        {
            watch.Restart();
            await graph.LoadAsync();
            timings.Add($"graph {graph.Name}: {graph.NodeCount} nodes, {graph.Ir!.Lines.Count} IR lines in {watch.ElapsedMilliseconds} ms");
            Assert.Equal(graph.NodeCount, graph.Nodes.Count);
        }

        var tabs = window.GetVisualDescendants().OfType<TabControl>().Single();
        for (int i = 0; i < tabs.ItemCount; i++)
        {
            watch.Restart();
            tabs.SelectedIndex = i;
            HeadlessApp.Render(window);
            timings.Add($"render tab {i} {watch.ElapsedMilliseconds} ms");
        }

        output?.WriteLine($"{path}: {package.Overview.FileSize}, {package.Overview.Sections.Count} sections, {package.Weights.Weights.Count} weights, "
            + $"{package.Artifacts.Sets.Count} artifact sets");
        foreach (var timing in timings)
        {
            output?.WriteLine("  " + timing);
        }

        window.Close();
    }
}
