namespace Sharplight.Graph.Optimization;

/// <summary>
/// Removes nodes whose outputs are not used by a graph output or by a node that is kept. Nodes with side effects (state writes),
/// state reads, data-dependent symbols and control flow are always kept; graph inputs are never removed.
/// </summary>
public sealed class DeadNodeElimination : IGraphPass
{
    /// <inheritdoc/>
    public string Name => "dead-nodes";

    /// <inheritdoc/>
    public bool RepeatUntilStable => false;

    /// <inheritdoc/>
    public ComputeGraph Run(ComputeGraph graph)
    {
        var live = new HashSet<Value>(graph.Outputs.Values);
        var rewriter = new GraphRewriter(graph);
        for (int k = graph.Nodes.Count - 1; k >= 0; k--)
        {
            var n = graph.Nodes[k];
            if (!GraphFacts.IsPinned(n, graph) && !n.Outputs.Any(live.Contains))
            {
                rewriter.Remove(n);
                continue;
            }

            live.UnionWith(n.Inputs);
            live.UnionWith(n.Writes);
        }

        return rewriter.Build();
    }
}
