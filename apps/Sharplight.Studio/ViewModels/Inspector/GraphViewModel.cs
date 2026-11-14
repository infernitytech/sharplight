using System.ComponentModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Graph;
using Sharplight.Package;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>
/// One stored graph: summary, inputs and outputs, op histogram, and three linked views of its nodes (visual graph, IR, node list).
/// The node list, the IR and the graph layout (possibly 100k+ entries each) are built on a background thread the first time the
/// graph is shown; the node filter runs on a background thread as well and stale results are dropped. Selecting a node in any view
/// selects it in the others and shows its details.
/// </summary>
internal sealed partial class GraphViewModel : ObservableObject
{
    private readonly GraphInfo _graph;
    private readonly IReadOnlyDictionary<string, PackageWeightInfo> _weights;
    private IReadOnlyList<NodeRow>? _allNodes;
    private Dictionary<int, NodeRow>? _nodesById;
    private Dictionary<int, NodeInfo>? _infoById;
    private Dictionary<int, List<int>>? _consumers;
    private Task? _loading;
    private int _filterVersion;

    /// <param name="graph">The stored graph.</param>
    /// <param name="weights">The package's weight index by name.</param>
    public GraphViewModel(GraphInfo graph, IReadOnlyDictionary<string, PackageWeightInfo> weights)
    {
        _graph = graph;
        _weights = weights;
        Inputs = [.. graph.Inputs.Select(GraphValueRow.From)];
        Outputs = [.. graph.Outputs.Select(GraphValueRow.From)];
        double total = Math.Max(1, graph.NodeCount);
        OpHistogram = [.. graph.OpHistogram.Select(kv => new OpCountRow(kv.Key, kv.Value, 100 * kv.Value / total))];
        NodeSummary = $"{graph.NodeCount:N0} nodes";
    }

    /// <summary>Graph name.</summary>
    public string Name => _graph.Name;

    /// <summary>Number of nodes.</summary>
    public int NodeCount => _graph.NodeCount;

    /// <summary>One line for the graph list.</summary>
    public string Summary => $"{_graph.NodeCount:N0} nodes";

    /// <summary>Nodes, weights and states of the graph.</summary>
    public string Stats =>
        $"{_graph.NodeCount:N0} nodes · {_graph.OpHistogram.Count} ops · {_graph.WeightCount:N0} weights ({ByteFormat.Format(_graph.WeightBytes)}) · {_graph.StateCount} states";

    /// <summary>Graph inputs.</summary>
    public IReadOnlyList<GraphValueRow> Inputs { get; }

    /// <summary>Graph outputs.</summary>
    public IReadOnlyList<GraphValueRow> Outputs { get; }

    /// <summary>Nodes per op, most frequent first.</summary>
    public IReadOnlyList<OpCountRow> OpHistogram { get; }

    /// <summary>The visible view.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsGraphMode), nameof(IsIrMode), nameof(IsNodesMode), nameof(ShowsDetails))]
    public partial GraphViewMode Mode { get; set; }

    /// <summary>The visual graph is shown.</summary>
    public bool IsGraphMode => Mode == GraphViewMode.Graph;

    /// <summary>The IR is shown.</summary>
    public bool IsIrMode => Mode == GraphViewMode.Ir;

    /// <summary>The node list is shown.</summary>
    public bool IsNodesMode => Mode == GraphViewMode.Nodes;

    /// <summary>The node details panel is shown (the IR view has its own navigation panel instead).</summary>
    public bool ShowsDetails => Mode != GraphViewMode.Ir;

    /// <summary>
    /// Filter of the node list: whitespace-separated terms, all must match op, name, group, shapes or attributes;
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

    /// <summary>The layout of the visual graph; null until <see cref="LoadAsync"/> completes.</summary>
    [ObservableProperty]
    public partial GraphLayout? Layout { get; private set; }

    /// <summary>True while the node list, IR and layout are built.</summary>
    [ObservableProperty]
    public partial bool IsLoading { get; private set; }

    /// <summary>"Showing n of m nodes".</summary>
    [ObservableProperty]
    public partial string NodeSummary { get; private set; }

    /// <summary>The selected node; its IR line, list row and graph box are selected too.</summary>
    [ObservableProperty]
    public partial NodeRow? SelectedNode { get; set; }

    /// <summary>Id of the selected node, -1 for none (bound to the graph view).</summary>
    [ObservableProperty]
    public partial int SelectedNodeId { get; set; } = -1;

    /// <summary>Details of the selected node.</summary>
    [ObservableProperty]
    public partial NodeDetailsViewModel? Details { get; private set; }

    /// <summary>The op histogram row last selected: filters the node list to that op and highlights its nodes in the graph.</summary>
    [ObservableProperty]
    public partial OpCountRow? SelectedOp { get; set; }

    /// <summary>The op whose nodes the graph view highlights, or null.</summary>
    public string? HighlightOp => SelectedOp?.Op;

    /// <summary>Text the graph search looks for (op, output name or group).</summary>
    [ObservableProperty]
    public partial string? NodeSearch { get; set; }

    /// <summary>Result of the last graph search.</summary>
    [ObservableProperty]
    public partial string SearchStatus { get; private set; } = "";

    /// <summary>Completes when the latest <see cref="NodeFilter"/> has been applied to <see cref="Nodes"/>.</summary>
    public Task Filtering { get; private set; } = Task.CompletedTask;

    /// <summary>Builds the node list, the IR and the layout (once) and applies the current filter. Call on the UI thread.</summary>
    public Task LoadAsync() => _loading ??= LoadCoreAsync();

    /// <summary>Selects a node by id (from the graph, the details or the search).</summary>
    [RelayCommand]
    public void GoToNode(int id)
    {
        if (_nodesById is not null && _nodesById.TryGetValue(id, out var node))
        {
            SelectedNode = node;
        }
    }

    [RelayCommand]
    private void SetMode(string mode)
    {
        if (Enum.TryParse<GraphViewMode>(mode, out var value))
        {
            Mode = value;
        }
    }

    [RelayCommand]
    private void ClearOp() => SelectedOp = null;

    /// <summary>Selects the next node (after the selected one) whose op, output names or group contain <see cref="NodeSearch"/>.</summary>
    [RelayCommand]
    private void FindNode() => Find(forward: true);

    /// <summary>Selects the previous matching node.</summary>
    [RelayCommand]
    private void FindPreviousNode() => Find(forward: false);

    partial void OnNodeFilterChanged(string? value) => Filtering = FilterAsync(value);

    partial void OnNodeSearchChanged(string? value) => SearchStatus = "";

    partial void OnSelectedOpChanged(OpCountRow? value)
    {
        NodeFilter = value is null ? null : NodeRow.OpTermPrefix + value.Op;
        OnPropertyChanged(nameof(HighlightOp));
    }

    partial void OnSelectedNodeIdChanged(int value)
    {
        if (value < 0)
        {
            SelectedNode = null;
        }
        else
        {
            GoToNode(value);
        }
    }

    partial void OnSelectedNodeChanged(NodeRow? value)
    {
        SelectedNodeId = value?.Id ?? -1;
        Details = value is null || _infoById is null || _consumers is null
            ? null
            : new NodeDetailsViewModel(value.Info, _infoById, _consumers.TryGetValue(value.Id, out var readers) ? readers : [], _weights);
        if (value is not null && Ir?.LineOfNode(value.Id) is { } line && !ReferenceEquals(Ir.SelectedLine, line))
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

    private void Find(bool forward)
    {
        if (string.IsNullOrWhiteSpace(NodeSearch) || _allNodes is not { Count: > 0 } all)
        {
            return;
        }

        var terms = TextFilter.Terms(NodeSearch);
        int start = SelectedNode is { } selected ? IndexOf(all, selected) : (forward ? -1 : 0);
        for (int step = 1; step <= all.Count; step++)
        {
            int index = (((forward ? start + step : start - step) % all.Count) + all.Count) % all.Count;
            var row = all[index];
            if (terms.All(term => TextFilter.MatchesTerm(term, row.Op, row.Name, row.Group)))
            {
                SelectedNode = row;
                SearchStatus = $"#{row.Id}";
                return;
            }
        }

        SearchStatus = "not found";
    }

    private static int IndexOf(IReadOnlyList<NodeRow> rows, NodeRow row)
    {
        for (int i = 0; i < rows.Count; i++)
        {
            if (ReferenceEquals(rows[i], row))
            {
                return i;
            }
        }

        return -1;
    }

    private async Task LoadCoreAsync()
    {
        IsLoading = true;
        NodeSummary = $"Loading {_graph.NodeCount:N0} nodes…";
        var graph = _graph;
        var built = await Task.Run(() =>
        {
            var infos = graph.Nodes;
            IReadOnlyList<NodeRow> rows = [.. infos.Select(NodeRow.From)];
            var consumers = new Dictionary<int, List<int>>();
            foreach (var node in infos)
            {
                foreach (var input in node.InputValues)
                {
                    if (input.Kind == ValueKind.Intermediate && input.Producer >= 0)
                    {
                        if (!consumers.TryGetValue(input.Producer, out var list))
                        {
                            list = [];
                            consumers.Add(input.Producer, list);
                        }

                        if (list.Count == 0 || list[^1] != node.Id)
                        {
                            list.Add(node.Id);
                        }
                    }
                }
            }

            var ir = IrViewModel.Build(graph);
            var layout = GraphLayout.Build(infos, graph.Inputs, graph.Outputs, graph.Ir);
            return (Rows: rows, Consumers: consumers, Ir: ir, Layout: layout);
        });
        _allNodes = built.Rows;
        _nodesById = built.Rows.ToDictionary(n => n.Id);
        _infoById = built.Rows.ToDictionary(n => n.Id, n => n.Info);
        _consumers = built.Consumers;
        built.Ir.PropertyChanged += OnIrPropertyChanged;
        Ir = built.Ir;
        Layout = built.Layout;
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
