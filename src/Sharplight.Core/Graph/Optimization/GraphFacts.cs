namespace Sharplight.Graph.Optimization;

/// <summary>Facts about nodes that decide whether a pass may remove, merge or evaluate them.</summary>
internal static class GraphFacts
{
    /// <summary>
    /// Whether the node must be kept even when its outputs are unused, and must not be merged with an equal node: it writes
    /// state, reads state (its result depends on when it runs relative to state writes), owns a data-dependent symbol that
    /// other shapes may refer to, or runs subgraphs.
    /// </summary>
    public static bool IsPinned(Node n, ComputeGraph graph)
        => n.Writes.Length > 0
           || n.Op.Category == OpCategory.ControlFlow
           || n.Inputs.Any(v => v.Kind == ValueKind.State)
           || OwnsDynamicSymbol(n, graph);

    /// <summary>Whether an attribute of the node names one of the graph's data-dependent symbols.</summary>
    public static bool OwnsDynamicSymbol(Node n, ComputeGraph graph)
    {
        if (graph.DynamicSymbols.Count == 0)
        {
            return false;
        }

        foreach (var value in n.Attributes.Values)
        {
            if (value is string s && graph.DynamicSymbols.Any(d => d.Key == s))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Whether every dimension of the shape is a fixed number.</summary>
    public static bool IsFullyStatic(Tensors.Shape shape) => shape.Dims.All(d => !d.IsSymbolic);
}
