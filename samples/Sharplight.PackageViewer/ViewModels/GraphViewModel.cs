using System.ComponentModel;
using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Package;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>
/// One stored graph: summary, inputs/outputs, op histogram, the node table and the IR view. Node table and IR (possibly 100k+ rows
/// each) are built on a background thread the first time the graph is shown; the node filter runs on a background thread as well and
/// stale results are dropped. Selecting a node selects its IR line and vice versa.
/// </summary>
public sealed partial class GraphViewModel : ObservableObject
{
    private readonly GraphInfo _graph;
    private IReadOnlyList<NodeRow>? _allNodes;
    private Dictionary<int, NodeRow>? _nodesById;
    private Task? _loading;
    private int _filterVersion;

    public GraphViewModel(GraphInfo graph)
    {
        _graph = graph;
        Inputs = [.. graph.Inputs.Select(GraphValueRow.From)];
        Outputs = [.. graph.Outputs.Select(GraphValueRow.From)];
        double total = Math.Max(1, graph.NodeCount);
        OpHistogram = [.. graph.OpHistogram.Select(kv => new OpCountRow(kv.Key, kv.Value, 100 * kv.Value / total))];
        NodeSummary = $"{graph.NodeCount:N0} nodes (not loaded)";
    }

    public string Name => _graph.Name;

    public int NodeCount => _graph.NodeCount;

    /// <summary>One line for the graph list.</summary>
    public string Summary => $"{_graph.NodeCount:N0} nodes, {_graph.OpHistogram.Count} ops";

    public string Details =>
        $"{_graph.NodeCount:N0} nodes, {_graph.WeightCount} weights ({ByteSizeFormat.Format(_graph.WeightBytes)}), {_graph.StateCount} state(s)";

    public IReadOnlyList<GraphValueRow> Inputs { get; }

    public IReadOnlyList<GraphValueRow> Outputs { get; }

    public IReadOnlyList<OpCountRow> OpHistogram { get; }

    /// <summary>
    /// Filter of the node table: whitespace-separated terms, all must match op, name, group, shapes or attributes;
    /// <c>op:Name</c> matches the op exactly.
    /// </summary>
    [ObservableProperty]
    public partial string? NodeFilter { get; set; }

    /// <summary>The nodes matching <see cref="NodeFilter"/>; empty until <see cref="LoadAsync"/> completes.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<NodeRow> Nodes { get; private set; } = [];

    /// <summary>The textual IR; null until <see cref="LoadAsync"/> completes.</summary>
    [ObservableProperty]
    public partial IrViewModel? Ir { get; private set; }

    [ObservableProperty]
    public partial bool IsLoading { get; private set; }

    /// <summary>"Showing n of m nodes".</summary>
    [ObservableProperty]
    public partial string NodeSummary { get; private set; }

    /// <summary>The selected node; its IR line is selected too.</summary>
    [ObservableProperty]
    public partial NodeRow? SelectedNode { get; set; }

    /// <summary>The op histogram row last selected; selecting one filters the node table to that op.</summary>
    [ObservableProperty]
    public partial OpCountRow? SelectedOp { get; set; }

    /// <summary>Completes when the latest <see cref="NodeFilter"/> has been applied to <see cref="Nodes"/>.</summary>
    public Task Filtering { get; private set; } = Task.CompletedTask;

    /// <summary>Builds the node table and the IR (once) and applies the current filter. Call on the UI thread.</summary>
    public Task LoadAsync() => _loading ??= LoadCoreAsync();

    partial void OnNodeFilterChanged(string? value) => Filtering = FilterAsync(value);

    partial void OnSelectedOpChanged(OpCountRow? value)
    {
        if (value is not null)
        {
            NodeFilter = NodeRow.OpTermPrefix + value.Op;
        }
    }

    partial void OnSelectedNodeChanged(NodeRow? value)
    {
        if (value is not null && Ir?.LineOfNode(value.Id) is { } line)
        {
            Ir.GoTo(line);
        }
    }

    private void OnIrPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (e.PropertyName == nameof(IrViewModel.SelectedLine) && Ir?.SelectedLine?.Line is { Scope: 0, NodeId: >= 0 } line
            && _nodesById is not null && _nodesById.TryGetValue(line.NodeId, out var node))
        {
            SelectedNode = node;
        }
    }

    private async Task LoadCoreAsync()
    {
        IsLoading = true;
        NodeSummary = $"Loading {_graph.NodeCount:N0} nodes...";
        var graph = _graph;
        var (nodes, ir) = await Task.Run(() => ((IReadOnlyList<NodeRow>)[.. graph.Nodes.Select(NodeRow.From)], IrViewModel.Build(graph)));
        _allNodes = nodes;
        _nodesById = nodes.ToDictionary(n => n.Id);
        ir.PropertyChanged += OnIrPropertyChanged;
        Ir = ir;
        IsLoading = false;
        Filtering = FilterAsync(NodeFilter);
        await Filtering;
    }

    private async Task FilterAsync(string? filter)
    {
        if (_allNodes is not { } all)
        {
            return;
        }

        int version = ++_filterVersion;
        var rows = TextFilter.Terms(filter).Length == 0 ? all : await Task.Run(() => TextFilter.Apply(all, filter));
        if (version != _filterVersion)
        {
            return;
        }

        Nodes = rows;
        NodeSummary = rows.Count == all.Count ? $"{all.Count:N0} nodes" : $"Showing {rows.Count:N0} of {all.Count:N0} nodes";
    }
}
