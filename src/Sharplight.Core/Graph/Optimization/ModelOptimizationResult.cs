using Sharplight.Models;

namespace Sharplight.Graph.Optimization;

/// <summary>The result of optimizing every graph of a model definition.</summary>
/// <param name="Definition">
/// The definition with optimized graphs and the <see cref="GraphOptimizer.OptimizedProperty"/> property set to the pipeline id; it
/// shares weights, states, metadata and owned resources with the input definition.
/// </param>
/// <param name="Statistics">One entry per pass run and graph, in execution order.</param>
/// <param name="NodesBefore">Node count of each top-level graph before optimization, by graph name.</param>
/// <param name="NodesAfter">Node count of each top-level graph after optimization, by graph name.</param>
/// <param name="Elapsed">Total wall time.</param>
public sealed record ModelOptimizationResult(ModelDefinition Definition, IReadOnlyList<GraphPassStatistics> Statistics,
    IReadOnlyDictionary<string, int> NodesBefore, IReadOnlyDictionary<string, int> NodesAfter, TimeSpan Elapsed);
