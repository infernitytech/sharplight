using Sharplight.Graph;
using Sharplight.Package;

namespace Sharplight.Studio.Services.Inspection;

/// <summary>
/// A layered (top to bottom) layout of a stored graph for the graph view: graph inputs on the first row, every node one row below
/// its deepest producer, graph outputs below their producer. Within a row nodes sit near the mean position of their producers, so
/// chains stay straight. Nodes that read only inputs or weights are moved down next to their first consumer. Weights, states and
/// constants are not drawn as boxes (the node details list them). Linear in the number of nodes and edges plus a sort per row, so
/// graphs with 100,000 nodes lay out in well under a second. Pure data: built off the UI thread.
/// </summary>
internal sealed class GraphLayout
{
    /// <summary>Width of an op box.</summary>
    public const double NodeWidth = 184;

    /// <summary>Height of an op box.</summary>
    public const double NodeHeight = 46;

    /// <summary>Width of an input or output box.</summary>
    public const double TerminalWidth = 164;

    /// <summary>Height of an input or output box.</summary>
    public const double TerminalHeight = 34;

    /// <summary>Vertical distance between rows.</summary>
    public const double RankPitch = NodeHeight + 38;

    /// <summary>Horizontal gap between boxes of a row.</summary>
    public const double Gap = 28;

    private readonly Dictionary<int, int> _indexOfNode;

    private GraphLayout(IReadOnlyList<LayoutNode> nodes, IReadOnlyList<LayoutEdge> edges, IReadOnlyList<IReadOnlyList<int>> predecessors,
        IReadOnlyList<IReadOnlyList<int>> successors, Dictionary<int, int> indexOfNode)
    {
        Nodes = nodes;
        Edges = edges;
        Predecessors = predecessors;
        Successors = successors;
        _indexOfNode = indexOfNode;
        if (nodes.Count > 0)
        {
            Left = nodes.Min(n => n.X);
            Right = nodes.Max(n => n.X + n.Width);
            Bottom = nodes.Max(n => n.Y + n.Height);
            RankCount = nodes.Max(n => n.Rank) + 1;
        }
    }

    /// <summary>Every box: inputs, nodes in execution order, outputs.</summary>
    public IReadOnlyList<LayoutNode> Nodes { get; }

    /// <summary>Every dataflow edge between boxes.</summary>
    public IReadOnlyList<LayoutEdge> Edges { get; }

    /// <summary>Per box, the boxes it reads from.</summary>
    public IReadOnlyList<IReadOnlyList<int>> Predecessors { get; }

    /// <summary>Per box, the boxes that read from it.</summary>
    public IReadOnlyList<IReadOnlyList<int>> Successors { get; }

    /// <summary>Leftmost x of any box.</summary>
    public double Left { get; }

    /// <summary>Rightmost x of any box.</summary>
    public double Right { get; }

    /// <summary>Lowest y of any box (the top is 0).</summary>
    public double Bottom { get; }

    /// <summary>Number of rows.</summary>
    public int RankCount { get; }

    /// <summary>The box of a node, by node id; null for an unknown id.</summary>
    public LayoutNode? NodeOf(int nodeId) => _indexOfNode.TryGetValue(nodeId, out int index) ? Nodes[index] : null;

    /// <summary>Lays out a stored graph.</summary>
    /// <param name="nodes">The nodes in execution order (<see cref="GraphInfo.Nodes"/>).</param>
    /// <param name="inputs">The graph inputs (<see cref="GraphInfo.Inputs"/>).</param>
    /// <param name="outputs">The graph outputs (<see cref="GraphInfo.Outputs"/>).</param>
    /// <param name="ir">The IR (<see cref="GraphInfo.Ir"/>), for the value ids of the inputs and outputs.</param>
    public static GraphLayout Build(IReadOnlyList<NodeInfo> nodes, IReadOnlyList<GraphValueInfo> inputs, IReadOnlyList<GraphValueInfo> outputs,
        IReadOnlyList<IrLine> ir)
    {
        var boxes = new List<LayoutNode>(inputs.Count + nodes.Count + outputs.Count);
        var preds = new List<List<int>>(boxes.Capacity);
        var boxOfValue = new Dictionary<int, int>();
        var indexOfNode = new Dictionary<int, int>(nodes.Count);

        // Inputs, in the order of the IR's input lines (which define their value ids).
        int inputLine = 0;
        foreach (var line in ir)
        {
            if (line.Scope == 0 && line.Kind == GraphDumpLineKind.Input && inputLine < inputs.Count && line.Defines.Count == 1)
            {
                var input = inputs[inputLine++];
                boxOfValue[line.Defines[0]] = boxes.Count;
                boxes.Add(Terminal(boxes.Count, LayoutNodeKind.Input, input));
                preds.Add([]);
            }
        }

        foreach (var node in nodes)
        {
            int index = boxes.Count;
            indexOfNode[node.Id] = index;
            var first = node.Outputs.Count > 0 ? node.Outputs[0] : null;
            boxes.Add(new LayoutNode
            {
                Index = index,
                NodeId = node.Id,
                Kind = LayoutNodeKind.Op,
                Title = node.Op,
                Subtitle = first is null ? "" : node.Outputs.Count > 1 ? $"{first.Type}{first.Shape} +{node.Outputs.Count - 1}" : $"{first.Type}{first.Shape}",
                Group = node.Group,
                Category = OpCategories.Of(node.Op),
                WeightCount = node.InputValues.Count(v => v.Kind == ValueKind.Weight),
                Width = NodeWidth,
                Height = NodeHeight,
            });
            var from = new List<int>();
            foreach (var value in node.InputValues)
            {
                int source = value.Kind switch
                {
                    ValueKind.Intermediate when value.Producer >= 0 && indexOfNode.TryGetValue(value.Producer, out int producer) => producer,
                    ValueKind.Input when boxOfValue.TryGetValue(value.ValueId, out int input) => input,
                    _ => -1,
                };
                if (source >= 0 && !from.Contains(source))
                {
                    from.Add(source);
                }
            }

            preds.Add(from);
        }

        // Value ids of node outputs come from the IR (node lines define them).
        foreach (var line in ir)
        {
            if (line.Scope == 0 && line.Kind == GraphDumpLineKind.Node && indexOfNode.TryGetValue(line.NodeId, out int index))
            {
                foreach (int value in line.Defines)
                {
                    boxOfValue[value] = index;
                }
            }
        }

        int outputLine = 0;
        foreach (var line in ir)
        {
            if (line.Scope == 0 && line.Kind == GraphDumpLineKind.Output && outputLine < outputs.Count)
            {
                var output = outputs[outputLine++];
                preds.Add(line.Uses.Count == 1 && boxOfValue.TryGetValue(line.Uses[0], out int source) ? [source] : []);
                boxes.Add(Terminal(boxes.Count, LayoutNodeKind.Output, output));
            }
        }

        var succs = new List<int>[boxes.Count];
        for (int i = 0; i < succs.Length; i++)
        {
            succs[i] = [];
        }

        for (int i = 0; i < boxes.Count; i++)
        {
            foreach (int p in preds[i])
            {
                succs[p].Add(i);
            }
        }

        AssignRanks(boxes, preds, succs);
        Place(boxes, preds);

        var edges = new List<LayoutEdge>();
        for (int i = 0; i < boxes.Count; i++)
        {
            var from = preds[i];
            for (int k = 0; k < from.Count; k++)
            {
                edges.Add(new LayoutEdge(from[k], i, (k + 1.0) / (from.Count + 1.0)));
            }
        }

        return new GraphLayout(boxes, edges, preds, succs, indexOfNode);
    }

    private static LayoutNode Terminal(int index, LayoutNodeKind kind, GraphValueInfo value) => new()
    {
        Index = index,
        NodeId = -1,
        Kind = kind,
        Title = value.Name,
        Subtitle = $"{value.Type}{value.Shape}",
        Width = TerminalWidth,
        Height = TerminalHeight,
    };

    /// <summary>Longest path from the inputs; boxes that read no node output move down next to their first consumer.</summary>
    private static void AssignRanks(List<LayoutNode> boxes, List<List<int>> preds, List<int>[] succs)
    {
        var fed = new bool[boxes.Count];
        for (int i = 0; i < boxes.Count; i++)
        {
            var box = boxes[i];
            if (box.Kind == LayoutNodeKind.Input)
            {
                box.Rank = 0;
                continue;
            }

            int rank = 1;
            foreach (int p in preds[i])
            {
                rank = Math.Max(rank, boxes[p].Rank + 1);
                fed[i] |= boxes[p].Kind != LayoutNodeKind.Input;
            }

            box.Rank = rank;
        }

        for (int i = boxes.Count - 1; i >= 0; i--)
        {
            var box = boxes[i];
            if (box.Kind != LayoutNodeKind.Op || fed[i] || succs[i].Count == 0)
            {
                continue;
            }

            int first = succs[i].Min(s => boxes[s].Rank);
            box.Rank = Math.Max(box.Rank, first - 1);
        }
    }

    /// <summary>Row by row: order the boxes by the mean position of their producers, then pack them without overlap.</summary>
    private static void Place(List<LayoutNode> boxes, List<List<int>> preds)
    {
        int ranks = boxes.Count == 0 ? 0 : boxes.Max(b => b.Rank) + 1;
        var rows = new List<int>[ranks];
        for (int r = 0; r < ranks; r++)
        {
            rows[r] = [];
        }

        foreach (var box in boxes)
        {
            rows[box.Rank].Add(box.Index);
        }

        var target = new double[boxes.Count];
        foreach (var row in rows)
        {
            if (row.Count == 0)
            {
                continue;
            }

            foreach (int i in row)
            {
                var from = preds[i];
                target[i] = from.Count == 0 ? 0 : from.Average(p => boxes[p].CenterX);
            }

            row.Sort((a, b) => target[a] != target[b] ? target[a].CompareTo(target[b]) : a.CompareTo(b));
            double right = double.NegativeInfinity;
            double displacement = 0;
            foreach (int i in row)
            {
                var box = boxes[i];
                double x = Math.Max(target[i] - (box.Width / 2), right + Gap);
                box.X = x;
                right = x + box.Width;
                displacement += target[i] - box.CenterX;
            }

            double shift = displacement / row.Count;
            foreach (int i in row)
            {
                var box = boxes[i];
                box.X += shift;
                box.Y = (box.Rank * RankPitch) + ((NodeHeight - box.Height) / 2);
            }
        }
    }
}
