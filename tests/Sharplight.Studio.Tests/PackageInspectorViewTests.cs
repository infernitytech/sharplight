using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Logging;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Sharplight.Studio.Controls;
using Sharplight.Studio.Models;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Services.Inspection;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Inspector;
using Sharplight.Studio.Views;
using Sharplight.Studio.Views.Inspector;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The package inspector inside the Studio window, rendered headlessly: every tab binds cleanly, a 100,000-node graph stays
/// virtualized and responsive, and (with <c>SHARPLIGHT_INSPECTOR_PACKAGES</c>, a list of <c>.slm</c> paths separated by the
/// platform's path separator) real packages render in both themes. With <c>SHARPLIGHT_STUDIO_SCREENSHOTS</c> set to a directory the
/// frames are saved as PNG files (used for docs/studio.md).
/// </summary>
public sealed class PackageInspectorViewTests
{
    private const int LargeNodeCount = 100_000;
    private const string PackagesVariable = "SHARPLIGHT_INSPECTOR_PACKAGES";

    private static readonly string? s_screenshots = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_SCREENSHOTS");

    [Fact]
    public Task EveryTab_RendersWithoutBindingErrors() => HeadlessApp.RunAsync(async () =>
    {
        using var tmp = new TempDir();
        var path = tmp.File("small.slm");
        TestPackages.WriteSmall(path);
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
        var theme = new ThemeService(Application.Current!);
        theme.Apply(AppTheme.Dark, "Violet");
        var main = new MainWindowViewModel(services.Services);
        var window = new MainWindow { DataContext = main, Width = 1440, Height = 900 };
        window.Show();
        try
        {
            await main.NavigateAsync(main.Models);
            var library = main.Models.Library;
            Assert.True(await library.OpenInspectorAsync(path, null));
            var inspector = library.Inspector!;
            await inspector.Package!.Graphs.SelectedGraph!.LoadAsync();
            await HeadlessApp.RenderAsync(window);
            var view = window.GetVisualDescendants().OfType<PackageInspectorView>().Single();
            Assert.True(view.IsEffectivelyVisible);
            Assert.DoesNotContain(window.GetVisualDescendants().OfType<LibraryView>(), l => l.IsEffectivelyVisible);

            foreach (var tab in Enum.GetValues<InspectorTab>())
            {
                inspector.SelectedTab = (int)tab;
                await HeadlessApp.RenderAsync(window);
            }

            inspector.SelectedTab = (int)InspectorTab.Graphs;
            var graph = inspector.Package.Graphs.SelectedGraph!;
            graph.SelectedNodeId = graph.Nodes.First(n => n.Op == "MatMul").Id;
            foreach (var mode in Enum.GetValues<GraphViewMode>())
            {
                graph.Mode = mode;
                await HeadlessApp.RenderAsync(window);
            }

            graph.Mode = GraphViewMode.Graph;
            await HeadlessApp.RenderAsync(window);
            var canvas = window.GetVisualDescendants().OfType<GraphCanvas>().Single(c => c.IsEffectivelyVisible);
            Assert.Same(graph.Layout, canvas.Layout);
            var silu = graph.Layout!.NodeOf(graph.Nodes.Single(n => n.Op == "Silu").Id)!;
            var point = canvas.ToScreen(silu.CenterX, silu.CenterY);
            canvas.SetCurrentValue(GraphCanvas.SelectedNodeIdProperty, canvas.HitTest(point)!.NodeId);
            await HeadlessApp.RenderAsync(window);
            Assert.Equal("Silu", graph.Details?.Op);

            theme.Apply(AppTheme.Light, "Violet");
            await HeadlessApp.RenderAsync(window);
            using (var frame = window.CaptureRenderedFrame())
            {
                Assert.NotNull(frame);
            }

            inspector.CloseCommand.Execute(null);
            await HeadlessApp.RenderAsync(window);
            Assert.Null(library.Inspector);
        }
        finally
        {
            theme.Apply(AppTheme.Dark, "Violet");
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }

        Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
    });

    [Fact]
    public Task LargeGraph_IsVirtualizedAndResponsive()
    {
        var output = TestContext.Current.TestOutputHelper;
        return HeadlessApp.RunAsync(async () =>
        {
            using var tmp = new TempDir();
            var path = tmp.File("large.slm");
            TestPackages.WriteLarge(path, LargeNodeCount);
            using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
            var main = new MainWindowViewModel(services.Services);
            var window = new MainWindow { DataContext = main, Width = 1440, Height = 900 };
            window.Show();
            try
            {
                await main.NavigateAsync(main.Models);
                var timings = new List<string>();
                var watch = Stopwatch.StartNew();
                Assert.True(await main.Models.Library.OpenInspectorAsync(path, null));
                timings.Add($"open {watch.ElapsedMilliseconds} ms");
                var inspector = main.Models.Library.Inspector!;
                var graph = inspector.Package!.Graphs.SelectedGraph!;

                watch.Restart();
                await graph.LoadAsync();
                timings.Add($"nodes + IR + layout {watch.ElapsedMilliseconds} ms");
                Assert.Equal(LargeNodeCount, graph.Nodes.Count);

                inspector.SelectedTab = (int)InspectorTab.Graphs;
                watch.Restart();
                await HeadlessApp.RenderAsync(window);
                timings.Add($"render graph view {watch.ElapsedMilliseconds} ms");

                graph.Mode = GraphViewMode.Nodes;
                watch.Restart();
                await HeadlessApp.RenderAsync(window);
                timings.Add($"render node list {watch.ElapsedMilliseconds} ms");
                var nodeList = window.GetVisualDescendants().OfType<ListBox>().Single(l => ReferenceEquals(l.ItemsSource, graph.Nodes));
                int realizedRows = nodeList.GetRealizedContainers().Count();
                Assert.InRange(realizedRows, 1, 500);

                graph.Mode = GraphViewMode.Ir;
                watch.Restart();
                await HeadlessApp.RenderAsync(window);
                timings.Add($"render IR {watch.ElapsedMilliseconds} ms");
                var irList = window.GetVisualDescendants().OfType<ListBox>().Single(l => ReferenceEquals(l.ItemsSource, graph.Ir!.Lines));
                int realizedLines = irList.GetRealizedContainers().Count();
                Assert.InRange(realizedLines, 1, 500);

                graph.Mode = GraphViewMode.Nodes;
                watch.Restart();
                graph.NodeFilter = "op:Silu layers.42";
                await graph.Filtering;
                await HeadlessApp.RenderAsync(window);
                timings.Add($"filter {watch.ElapsedMilliseconds} ms");
                Assert.Equal(LargeNodeCount / 100 / 2, graph.Nodes.Count);

                graph.Mode = GraphViewMode.Graph;
                watch.Restart();
                graph.SelectedNode = graph.Nodes[^1];
                await HeadlessApp.RenderAsync(window);
                timings.Add($"select node, center graph, jump to IR {watch.ElapsedMilliseconds} ms");
                Assert.Equal(graph.SelectedNode.Id, graph.Ir!.SelectedLine!.Line.NodeId);
                Assert.NotNull(graph.Details);

                var canvas = window.GetVisualDescendants().OfType<GraphCanvas>().Single(c => c.IsEffectivelyVisible);
                watch.Restart();
                canvas.FitAll();
                await HeadlessApp.RenderAsync(window);
                timings.Add($"fit whole graph {watch.ElapsedMilliseconds} ms");

                output?.WriteLine($"{LargeNodeCount:N0} nodes: {string.Join(", ", timings)}; realized {realizedRows} list rows, {realizedLines} IR lines");
            }
            finally
            {
                window.Close();
                Dispatcher.UIThread.RunJobs();
            }
        });
    }

    [Fact]
    public Task RealPackages_RenderInBothThemes()
    {
        var paths = (Environment.GetEnvironmentVariable(PackagesVariable) ?? "")
            .Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries).Where(File.Exists).ToList();
        if (paths.Count == 0)
        {
            Assert.Skip($"{PackagesVariable} names no package");
        }

        var output = TestContext.Current.TestOutputHelper;
        return HeadlessApp.RunAsync(async () =>
        {
            var sink = new BindingErrorSink();
            Logger.Sink = sink;
            using var services = new TestServices(dispatcher: new AvaloniaDispatcher());
            var theme = new ThemeService(Application.Current!);
            var main = new MainWindowViewModel(services.Services);
            var window = new MainWindow { DataContext = main, Width = 1440, Height = 900 };
            window.Show();
            try
            {
                await main.NavigateAsync(main.Models);
                for (int p = 0; p < paths.Count; p++)
                {
                    var watch = Stopwatch.StartNew();
                    Assert.True(await main.Models.Library.OpenInspectorAsync(paths[p], null), main.Models.Library.Inspector?.Error);
                    var inspector = main.Models.Library.Inspector!;
                    long open = watch.ElapsedMilliseconds;
                    var graph = inspector.Package!.Graphs.SelectedGraph!;
                    watch.Restart();
                    await graph.LoadAsync();
                    output?.WriteLine($"{Path.GetFileName(Path.GetDirectoryName(paths[p]))}: {inspector.Status}; open {open} ms; graph {graph.Name} "
                        + $"({graph.NodeCount:N0} nodes, {graph.Ir!.Lines.Count:N0} IR lines) in {watch.ElapsedMilliseconds} ms; "
                        + $"{inspector.Package.Weights.Weights.Count:N0} weights, {inspector.Package.Artifacts.Sets.Count} artifact sets, "
                        + $"{inspector.Package.Sections.Rows.Count} regions");
                    Assert.Null(inspector.Package.Overview.DefinitionError);

                    // Select a node in the second layer, as a user exploring the graph would.
                    var node = graph.Nodes.FirstOrDefault(n => n.Op.Contains("Attention", StringComparison.Ordinal)) ?? graph.Nodes[graph.Nodes.Count / 2];
                    graph.SelectedNode = node;
                    var weights = inspector.Package.Weights;
                    weights.SelectedWeight = weights.Weights.FirstOrDefault();
                    var sections = inspector.Package.Sections;
                    sections.SelectedRow = sections.Rows.FirstOrDefault(r => r.Kind == "ARTS") ?? sections.SelectedRow;

                    foreach (var (variant, accent) in new[] { (AppTheme.Dark, "Violet"), (AppTheme.Light, "Violet") })
                    {
                        theme.Apply(variant, accent);
                        foreach (var tab in Enum.GetValues<InspectorTab>())
                        {
                            inspector.SelectedTab = (int)tab;
                            watch.Restart();
                            await HeadlessApp.RenderAsync(window);
                            if (tab == InspectorTab.Graphs)
                            {
                                window.GetVisualDescendants().OfType<GraphCanvas>().Single(c => c.IsEffectivelyVisible).CenterOnSelection();
                                await HeadlessApp.RenderAsync(window);
                            }

                            Save(window, $"inspector-{p}-{tab.ToString().ToLowerInvariant()}-{variant.ToString().ToLowerInvariant()}");
                            output?.WriteLine($"  {variant} {tab}: rendered in {watch.ElapsedMilliseconds} ms");
                        }
                    }

                    theme.Apply(AppTheme.Dark, "Violet");
                    inspector.CloseCommand.Execute(null);
                }
            }
            finally
            {
                window.Close();
                Dispatcher.UIThread.RunJobs();
                Logger.Sink = null;
            }

            Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
        });
    }

    private static void Save(Window window, string name)
    {
        using var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);
        if (s_screenshots is null)
        {
            return;
        }

        Directory.CreateDirectory(s_screenshots);
        frame.Save(Path.Combine(s_screenshots, $"studio-{name}.png"), new PngBitmapEncoderOptions());
    }
}
