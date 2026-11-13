namespace Sharplight.Graph.Optimization;

/// <summary>The result of optimizing one graph.</summary>
/// <param name="Graph">The optimized graph (the input graph itself when nothing changed).</param>
/// <param name="Statistics">One entry per pass run, in execution order, including those of optimized subgraphs.</param>
public sealed record GraphOptimizationResult(ComputeGraph Graph, IReadOnlyList<GraphPassStatistics> Statistics);
