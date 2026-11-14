using Sharplight.Graph;
using Sharplight.Package;
using Sharplight.Studio.Services.Inspection;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Inspector;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.Tests;

/// <summary>The package inspector's view models over packages written by <see cref="ModelPackage.Save"/>.</summary>
public sealed class PackageInspectorTests
{
    [Fact]
    public async Task SmallPackage_PopulatesEveryTab()
    {
        using var tmp = new TempDir();
        var path = tmp.File("small.slm");
        TestPackages.WriteSmall(path);
        var inspector = new PackageInspectorViewModel(path, null, () => { });
        Assert.True(await inspector.LoadAsync(), inspector.Error);
        var package = inspector.Package!;
        Assert.Equal("small", inspector.Title);
        Assert.True(inspector.IsLoaded);

        var overview = package.Overview;
        Assert.Null(overview.DefinitionError);
        Assert.Contains(overview.Package, f => f.Label == "Format version" && f.Value == ModelPackage.Version.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Assert.Contains(overview.Model, f => f.Label == "Architecture" && f.Value == "test_arch");
        Assert.Contains(overview.Model, f => f.Label == "Context length" && f.Value == "4096");
        Assert.Contains(overview.Tokenizer, f => f.Label == "Vocabulary" && f.Value == "3 tokens");
        Assert.Contains(overview.Tokenizer, f => f.Label == "End of sequence" && f.Value == "2 <eos>");
        Assert.True(overview.HasChatTemplate);
        Assert.Contains("messages", overview.ChatTemplate, StringComparison.Ordinal);
        Assert.Equal("kc", Assert.Single(overview.States).Name);
        Assert.Contains(overview.KindTotals, k => k.Detail == "ARTS × 2");

        var metadata = package.Metadata;
        Assert.Equal(7, metadata.Count);
        metadata.Filter = "tokenizer.tokens";
        var tokens = Assert.Single(metadata.Values);
        Assert.Equal(MetadataViewModel.MetadataSource, tokens.Source);
        Assert.Contains("a, b, <eos>", tokens.Detail, StringComparison.Ordinal);

        var graph = Assert.Single(package.Graphs.Graphs);
        Assert.Same(graph, package.Graphs.SelectedGraph);
        await graph.LoadAsync();
        Assert.Equal(graph.NodeCount, graph.Nodes.Count);
        Assert.NotNull(graph.Layout);
        Assert.Contains(graph.OpHistogram, o => o.Op == "MatMul" && o.Count == 2);
        graph.SelectedOp = graph.OpHistogram.Single(o => o.Op == "Silu");
        await graph.Filtering;
        Assert.Equal("Silu", Assert.Single(graph.Nodes).Op);
        Assert.Equal("Silu", graph.HighlightOp);
        graph.ClearOpCommand.Execute(null);
        await graph.Filtering;
        Assert.Equal(graph.NodeCount, graph.Nodes.Count);

        var weights = package.Weights;
        Assert.Equal(2, weights.Weights.Count);
        Assert.Equal(["f32", "f16"], weights.TypeTotals.Select(t => t.Label));
        weights.SortBy(nameof(WeightColumn.Size));
        Assert.Equal("w0", weights.Weights[0].Name);
        Assert.EndsWith("↓", weights.SizeHeader, StringComparison.Ordinal);
        weights.SortBy(nameof(WeightColumn.Size));
        Assert.Equal("w1", weights.Weights[0].Name);
        Assert.Equal(16, weights.Weights[0].BitsPerElement);
        weights.Filter = "w0";
        Assert.Equal("w0", Assert.Single(weights.Weights).Name);

        var artifacts = package.Artifacts;
        Assert.Equal(2, artifacts.Sets.Count);
        Assert.Equal(["binary", "code"], artifacts.Sets.Select(s => s.Tier));
        var code = artifacts.Sets.Single(s => s.Tier == "code");
        Assert.Contains(code.Descriptor, d => d.Label == "Code target" && d.Value == "spirv_1_6");
        Assert.Contains("subgroup size 32", code.BuiltFor, StringComparison.Ordinal);
        Assert.Equal(2, code.EntryCount);
        var kernel = code.Entries.Single(e => e.Key == TestPackages.CodeKernelKey);
        Assert.Equal("00112233445566778899aabbccddeeff", kernel.Hash);
        Assert.Equal("kernel/", kernel.Prefix);
        Assert.InRange(kernel.Offset, code.SectionOffset, code.SectionOffset + code.SectionLength);
        code.SortBy("Size");
        Assert.Equal(64, code.Entries[0].Bytes);
        code.EntryFilter = "ffee";
        Assert.Single(code.Entries);
        var binary = artifacts.Sets.Single(s => s.Tier == "binary");
        Assert.Equal("", Assert.Single(binary.Entries).Hash);

        var sections = package.Sections;
        Assert.Equal(SectionViews.HeaderKind, sections.Rows[0].Kind);
        Assert.Contains(sections.Rows, r => r.Kind == SectionViews.TableKind);
        Assert.Equal(["ARTS", "ARTS", "DEFN", "WDAT", "WIDX"],
            sections.Rows.Where(r => ModelPackageReader.SectionKinds.Contains(r.Kind)).Select(r => r.Kind).Order(StringComparer.Ordinal));
        Assert.Equal(package.Info.FileSize, sections.Rows.Sum(r => r.Length));
        for (int i = 1; i < sections.Rows.Count; i++)
        {
            Assert.Equal(sections.Rows[i - 1].Offset + sections.Rows[i - 1].Length, sections.Rows[i].Offset);
        }

        inspector.ShowSection(sections.Rows.First(r => r.Kind == "ARTS" && r.Offset == code.SectionOffset));
        Assert.Equal((int)InspectorTab.Artifacts, inspector.SelectedTab);
        Assert.Same(code, artifacts.SelectedSet);
        inspector.ShowSection(sections.Rows.Single(r => r.Kind == "WIDX"));
        Assert.Equal((int)InspectorTab.Weights, inspector.SelectedTab);
    }

    [Fact]
    public void EverySectionKindOfTheFormat_HasAView()
    {
        Assert.NotEmpty(ModelPackageReader.SectionKinds);
        foreach (var kind in ModelPackageReader.SectionKinds)
        {
            var view = SectionViews.For(kind);
            Assert.True(view.IsDecoded, kind);
            Assert.NotEqual(InspectorTab.Sections, view.Tab);
            Assert.DoesNotContain("Unknown", view.Title, StringComparison.Ordinal);
        }

        foreach (var kind in new[] { SectionViews.HeaderKind, SectionViews.TableKind, SectionViews.UnusedKind })
        {
            Assert.True(SectionViews.For(kind).IsDecoded, kind);
        }

        Assert.False(SectionViews.For("77").IsDecoded);
    }

    [Fact]
    public async Task UnknownSection_IsListedRawAndTheRestStillReads()
    {
        using var tmp = new TempDir();
        var path = tmp.File("future.slm");
        TestPackages.WriteSmall(path);
        TestPackages.AppendUnknownSection(path, 99, "future.data", 5000);
        var inspector = new PackageInspectorViewModel(path, null, () => { });
        Assert.True(await inspector.LoadAsync(), inspector.Error);
        var sections = inspector.Package!.Sections;
        var unknown = Assert.Single(sections.Rows, r => r.Kind == "99");
        Assert.False(unknown.IsDecoded);
        Assert.False(unknown.CanShow);
        Assert.Equal("future.data", unknown.Name);
        Assert.Equal(5000, unknown.Length);
        Assert.Contains(sections.Rows, r => r.Kind == SectionViews.UnusedKind);
        Assert.Equal(inspector.Package.Info.FileSize, sections.Rows.Sum(r => r.Length));
        Assert.Equal(2, inspector.Package.Artifacts.Sets.Count);
        Assert.Single(inspector.Package.Graphs.Graphs);
    }

    [Fact]
    public async Task GraphSelection_LinksLayoutIrAndDetails()
    {
        using var tmp = new TempDir();
        var path = tmp.File("small.slm");
        TestPackages.WriteSmall(path);
        var inspector = new PackageInspectorViewModel(path, null, () => { });
        Assert.True(await inspector.LoadAsync());
        var graph = inspector.Package!.Graphs.SelectedGraph!;
        await graph.LoadAsync();
        var ir = graph.Ir!;

        Assert.Equal(GraphDumpLineKind.Graph, ir.Lines[0].Line.Kind);
        Assert.Contains(ir.Lines, l => l.Line.Kind == GraphDumpLineKind.Weight && l.Text.Contains("w0", StringComparison.Ordinal));
        Assert.Contains(ir.Lines, l => l.Line.Kind == GraphDumpLineKind.State && l.Text.Contains("KeyCache", StringComparison.Ordinal));

        // Graph view selection (by id) -> node, details, IR line.
        var silu = graph.Nodes.Single(n => n.Op == "Silu");
        graph.SelectedNodeId = silu.Id;
        Assert.Same(silu, graph.SelectedNode);
        Assert.Equal(silu.Id, ir.SelectedLine!.Line.NodeId);
        var details = graph.Details!;
        Assert.Equal("Silu", details.Op);
        Assert.Equal("layers.1", details.Group);
        var input = Assert.Single(details.Inputs);
        Assert.Equal("node", input.Kind);
        Assert.True(input.HasProducer);

        // Follow the input to its producer (the second MatMul): its details list the weight and Silu as a consumer.
        graph.GoToNodeCommand.Execute(input.Producer);
        Assert.Equal("MatMul", graph.SelectedNode!.Op);
        Assert.Equal(graph.SelectedNode.Id, graph.SelectedNodeId);
        Assert.Contains(graph.Details!.Inputs, i => i.Kind == "weight" && i.Name == "w1" && i.Detail.Length > 0);
        Assert.Contains(graph.Details.Consumers, c => c.Id == silu.Id);
        Assert.Equal("1 weight, 32 B", graph.Details.WeightSummary);

        // IR navigation selects the node again.
        ir.BackCommand.Execute(null);
        Assert.Same(silu, graph.SelectedNode);
        var producer = Assert.Single(ir.Producers);
        ir.GoTo(producer.Target);
        Assert.Equal("MatMul", graph.SelectedNode!.Op);
        Assert.Contains(ir.Neighborhood, n => n.Target.Line.Kind == GraphDumpLineKind.Weight);

        ir.SearchText = "output y";
        ir.FindNextCommand.Execute(null);
        Assert.Equal(GraphDumpLineKind.Output, ir.SelectedLine!.Line.Kind);
        ir.SearchText = "no such text";
        ir.FindNextCommand.Execute(null);
        Assert.Equal("not found", ir.SearchStatus);

        // Graph search walks the matches; deselecting clears the details.
        graph.NodeSearch = "layers.1";
        graph.FindNodeCommand.Execute(null);
        Assert.Equal("layers.1", graph.SelectedNode!.Group);
        graph.SelectedNodeId = -1;
        Assert.Null(graph.SelectedNode);
        Assert.Null(graph.Details);

        graph.SetModeCommand.Execute("Ir");
        Assert.True(graph.IsIrMode);
        Assert.False(graph.ShowsDetails);
    }

    [Fact]
    public async Task MissingFile_ReportsAnError()
    {
        var inspector = new PackageInspectorViewModel(Path.Combine(Path.GetTempPath(), "does-not-exist.slm"), null, () => { });
        Assert.False(await inspector.LoadAsync());
        Assert.Null(inspector.Package);
        Assert.True(inspector.HasError);
        Assert.StartsWith("Cannot read", inspector.Error, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Library_OpensAndClosesTheInspector()
    {
        using var tmp = new TempDir();
        var path = tmp.File("model.slm");
        TestPackages.WriteSmall(path);
        using var services = new TestServices();
        services.Library.Entries.Add(new LibraryEntry { Id = "small", DisplayName = "small.slm", Path = path, Format = "SLM" });
        services.Library.Entries.Add(new LibraryEntry { Id = "other", DisplayName = "other.gguf", Path = tmp.File("other.gguf"), Format = "GGUF" });
        var library = new LibraryViewModel(services.Services, new FakeNavigator());
        await library.RefreshAsync();
        Assert.True(library.Entries[0].IsPackage);
        Assert.False(library.Entries[1].IsPackage);
        Assert.Same(library, library.Entries[0].Library);

        await library.InspectCommand.ExecuteAsync(library.Entries[0]);
        Assert.NotNull(library.Inspector);
        Assert.True(library.Inspector.IsLoaded);
        library.Inspector.CloseCommand.Execute(null);
        Assert.Null(library.Inspector);

        services.Platform.PickedFiles = [path];
        await library.OpenPackageFileCommand.ExecuteAsync(null);
        Assert.Equal(path, library.Inspector?.Path);
    }
}
