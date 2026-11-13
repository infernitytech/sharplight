using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Package;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>
/// The textual IR of one graph (the <c>--dump-graph</c> format, including control-flow subgraphs) with search and navigation:
/// the producers of the values the selected line reads, the consumers of the values it defines, a small dataflow neighborhood,
/// and a back history. Built off the UI thread with <see cref="Build"/>; use on the UI thread afterwards.
/// </summary>
public sealed partial class IrViewModel : ObservableObject
{
    /// <summary>Levels of producers and consumers shown in the neighborhood.</summary>
    public const int NeighborhoodLevels = 2;

    private const int MaxNeighborsPerLevel = 24;
    private const int MaxHistory = 256;

    private readonly Dictionary<(int Scope, int Value), int> _producers = [];
    private readonly Dictionary<(int Scope, int Value), List<int>> _consumers = [];
    private readonly Dictionary<int, int> _topLevelNodeLines = [];
    private readonly Stack<IrLineRow> _history = new();
    private bool _navigatingBack;

    private IrViewModel(IReadOnlyList<IrLine> lines)
    {
        var rows = new IrLineRow[lines.Count];
        for (int i = 0; i < lines.Count; i++)
        {
            var line = lines[i];
            rows[i] = new IrLineRow(i, new string(' ', 2 * line.Depth) + line.Text, line);
            foreach (int value in line.Defines)
            {
                _producers.TryAdd((line.Scope, value), i);
            }

            foreach (int value in line.Uses)
            {
                if (!_consumers.TryGetValue((line.Scope, value), out var consumers))
                {
                    consumers = [];
                    _consumers.Add((line.Scope, value), consumers);
                }

                consumers.Add(i);
            }

            if (line.Scope == 0 && line.NodeId >= 0)
            {
                _topLevelNodeLines[line.NodeId] = i;
            }
        }

        Lines = rows;
    }

    public IReadOnlyList<IrLineRow> Lines { get; }

    [ObservableProperty]
    public partial IrLineRow? SelectedLine { get; set; }

    /// <summary>For each value the selected line reads: the line that declares or produces it.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<IrReferenceRow> Producers { get; private set; } = [];

    /// <summary>For each value the selected line defines: the lines that read it.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<IrReferenceRow> Consumers { get; private set; } = [];

    /// <summary>Producers and consumers of the selected line up to <see cref="NeighborhoodLevels"/> levels, as an indented list.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<IrReferenceRow> Neighborhood { get; private set; } = [];

    /// <summary>Text to find (case-insensitive substring).</summary>
    [ObservableProperty]
    public partial string? SearchText { get; set; }

    [ObservableProperty]
    public partial string SearchStatus { get; private set; } = "";

    public static IrViewModel Build(GraphInfo graph) => new(graph.Ir);

    /// <summary>The line of a node of the stored graph (scope 0), by node id.</summary>
    public IrLineRow? LineOfNode(int nodeId) => _topLevelNodeLines.TryGetValue(nodeId, out int index) ? Lines[index] : null;

    /// <summary>Selects a line, remembering the current one for <see cref="BackCommand"/>.</summary>
    public void GoTo(IrLineRow? target)
    {
        if (target is not null)
        {
            SelectedLine = target;
        }
    }

    [RelayCommand]
    private void Back()
    {
        if (_history.TryPop(out var previous))
        {
            _navigatingBack = true;
            SelectedLine = previous;
            _navigatingBack = false;
        }
    }

    [RelayCommand]
    private void FindNext() => Find(forward: true);

    [RelayCommand]
    private void FindPrevious() => Find(forward: false);

    partial void OnSelectedLineChanged(IrLineRow? oldValue, IrLineRow? newValue)
    {
        if (oldValue is not null && !_navigatingBack)
        {
            _history.Push(oldValue);
            if (_history.Count > MaxHistory)
            {
                var kept = _history.Take(MaxHistory / 2).Reverse().ToList();
                _history.Clear();
                kept.ForEach(_history.Push);
            }
        }

        if (newValue is null)
        {
            Producers = [];
            Consumers = [];
            Neighborhood = [];
            return;
        }

        var line = newValue.Line;
        Producers = [.. line.Uses.Select(v => (Value: v, Line: Producer(line.Scope, v))).Where(p => p.Line is not null)
            .Select(p => new IrReferenceRow($"%{p.Value} from", p.Line!))];
        Consumers = [.. line.Defines.SelectMany(v => ConsumersOf(line.Scope, v).Select(c => new IrReferenceRow($"%{v} used by", c)))];
        Neighborhood = BuildNeighborhood(newValue);
    }

    partial void OnSearchTextChanged(string? value) => SearchStatus = "";

    private IrLineRow? Producer(int scope, int value) => _producers.TryGetValue((scope, value), out int index) ? Lines[index] : null;

    private IEnumerable<IrLineRow> ConsumersOf(int scope, int value) =>
        _consumers.TryGetValue((scope, value), out var lines) ? lines.Select(i => Lines[i]) : [];

    private IReadOnlyList<IrReferenceRow> BuildNeighborhood(IrLineRow center)
    {
        var rows = new List<IrReferenceRow>();
        AddLevels(rows, center, "<", row => row.Line.Uses.Select(v => Producer(row.Line.Scope, v)).OfType<IrLineRow>(), descending: true);
        rows.Add(new IrReferenceRow("*", center));
        AddLevels(rows, center, ">", row => row.Line.Defines.SelectMany(v => ConsumersOf(row.Line.Scope, v)), descending: false);
        return rows;
    }

    /// <summary>Breadth-first levels of producers (listed farthest first, above the center) or consumers (nearest first, below it).</summary>
    private static void AddLevels(List<IrReferenceRow> rows, IrLineRow center, string marker, Func<IrLineRow, IEnumerable<IrLineRow>> next, bool descending)
    {
        var levels = new List<List<IrLineRow>>();
        var seen = new HashSet<int> { center.Index };
        var frontier = new List<IrLineRow> { center };
        for (int level = 0; level < NeighborhoodLevels && frontier.Count > 0; level++)
        {
            frontier = [.. frontier.SelectMany(next).Where(r => seen.Add(r.Index)).Take(MaxNeighborsPerLevel)];
            levels.Add(frontier);
        }

        for (int i = 0; i < levels.Count; i++)
        {
            int level = descending ? levels.Count - 1 - i : i;
            var label = new string(marker[0], level + 1);
            rows.AddRange(levels[level].OrderBy(r => r.Index).Select(r => new IrReferenceRow(label, r)));
        }
    }

    private void Find(bool forward)
    {
        if (string.IsNullOrEmpty(SearchText) || Lines.Count == 0)
        {
            return;
        }

        int start = SelectedLine?.Index ?? (forward ? -1 : 0);
        for (int step = 1; step <= Lines.Count; step++)
        {
            int index = ((forward ? start + step : start - step) % Lines.Count + Lines.Count) % Lines.Count;
            if (Lines[index].Text.Contains(SearchText, StringComparison.OrdinalIgnoreCase))
            {
                GoTo(Lines[index]);
                SearchStatus = $"line {index + 1:N0}";
                return;
            }
        }

        SearchStatus = "not found";
    }
}
