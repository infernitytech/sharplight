namespace Sharplight.Graph.Optimization;

/// <summary>
/// A device-independent rewrite of a <see cref="ComputeGraph"/> that preserves its results exactly: the same inputs, outputs,
/// state effects and data-dependent symbols, computing bit-identical values on the reference (CPU) backend.
/// </summary>
/// <remarks>Implementations must be stateless or thread-safe: one pass instance may run on several graphs concurrently.</remarks>
public interface IGraphPass
{
    /// <summary>Short, stable name used in statistics and logs (for example <c>dead-nodes</c>).</summary>
    public string Name { get; }

    /// <summary>
    /// Whether the <see cref="GraphOptimizer"/> should run another round of the whole pipeline after this pass changed the graph,
    /// because its result can enable further rewrites (by itself or by the other passes).
    /// </summary>
    public bool RepeatUntilStable { get; }

    /// <summary>Rewrites a graph.</summary>
    /// <param name="graph">The graph to optimize; it is not modified.</param>
    /// <returns>The rewritten graph, or <paramref name="graph"/> itself when the pass changed nothing.</returns>
    public ComputeGraph Run(ComputeGraph graph);
}
