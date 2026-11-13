using Sharplight.Graph;

namespace Sharplight.Package;

/// <summary>Summary of one stored graph. The node table (<see cref="Nodes"/>) is built on first access.</summary>
public sealed class GraphInfo
{
    private readonly Lazy<IReadOnlyList<NodeInfo>> _nodes;
    private readonly Lazy<IReadOnlyList<IrLine>> _ir;

    internal GraphInfo(string name, ComputeGraph graph)
    {
        Name = name;
        NodeCount = graph.Nodes.Count;
        OpHistogram = [.. graph.Nodes.GroupBy(n => n.Op.Name, StringComparer.Ordinal)
            .Select(g => new KeyValuePair<string, int>(g.Key, g.Count()))
            .OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key, StringComparer.Ordinal)];
        Inputs = [.. graph.Inputs.Select(v => new GraphValueInfo(v.Name, v.Type, v.Shape, v.Input?.Role.ToString() ?? ""))];
        Outputs = [.. graph.Outputs.Select(o => new GraphValueInfo(o.Key, o.Value.Type, o.Value.Shape, ""))];
        var weights = graph.Weights.ToList();
        WeightCount = weights.Count;
        WeightBytes = weights.Sum(v => v.Weight!.ByteSize);
        StateCount = graph.States.Count();
        _nodes = new(() => [.. graph.Nodes.Select(Describe)]);
        _ir = new(() => IrLine.Build(graph));
    }

    /// <summary>Graph name (e.g. <c>decode</c>, <c>vision_encode</c>, <c>main</c>).</summary>
    public string Name { get; }

    /// <summary>Number of nodes.</summary>
    public int NodeCount { get; }

    /// <summary>Nodes per op, most frequent first.</summary>
    public IReadOnlyList<KeyValuePair<string, int>> OpHistogram { get; }

    /// <summary>Graph inputs in order.</summary>
    public IReadOnlyList<GraphValueInfo> Inputs { get; }

    /// <summary>Graph outputs.</summary>
    public IReadOnlyList<GraphValueInfo> Outputs { get; }

    /// <summary>Number of distinct weights the graph reads.</summary>
    public int WeightCount { get; }

    /// <summary>Stored bytes of those weights.</summary>
    public long WeightBytes { get; }

    /// <summary>Number of distinct states (KV caches, recurrent state) the graph uses.</summary>
    public int StateCount { get; }

    /// <summary>Every node in execution order; built on first access.</summary>
    public IReadOnlyList<NodeInfo> Nodes => _nodes.Value;

    /// <summary>
    /// The textual IR, one entry per line of <see cref="ComputeGraph.Dump"/> (inputs, weights, states, constants, nodes, control-flow
    /// subgraphs, outputs), with value references for navigation; built on first access, reads no weight data. Thread-safe.
    /// </summary>
    public IReadOnlyList<IrLine> Ir => _ir.Value;

    private static NodeInfo Describe(Node node) => new(
        node.Id,
        node.Op.Name,
        node.Group,
        node.Attributes.ToString(),
        [.. node.Inputs.Select(v => v.ToString())],
        [.. node.Outputs.Select(v => new GraphValueInfo(v.Name, v.Type, v.Shape, ""))]);
}
