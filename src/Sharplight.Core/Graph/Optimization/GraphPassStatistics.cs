namespace Sharplight.Graph.Optimization;

/// <summary>What one pass did to one graph in one round of a <see cref="GraphOptimizer"/> run.</summary>
/// <param name="Graph">The graph name; subgraphs (branches of control-flow nodes) are named <c>parent/node#id.attribute</c>.</param>
/// <param name="Pass">The pass name (<see cref="IGraphPass.Name"/>).</param>
/// <param name="Round">The pipeline round, starting at 1.</param>
/// <param name="NodesBefore">Node count before the pass.</param>
/// <param name="NodesAfter">Node count after the pass.</param>
/// <param name="Changed">Whether the pass changed the graph (it may rewrite nodes without changing their count).</param>
/// <param name="Elapsed">Wall time of the pass.</param>
public sealed record GraphPassStatistics(string Graph, string Pass, int Round, int NodesBefore, int NodesAfter, bool Changed, TimeSpan Elapsed);
