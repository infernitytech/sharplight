using Sharplight.Graph.Optimization;

namespace Sharplight.Package;

/// <summary>The result of <see cref="PackageOptimizer.Optimize"/>.</summary>
/// <param name="PackagePath">The package that was written (or, when nothing was done, the unchanged input package).</param>
/// <param name="Optimized">
/// Whether the pipeline ran and a package was written; <see langword="false"/> when the package already carried the pipeline's id
/// and <see cref="PackageOptimizeOptions.Force"/> was off.
/// </param>
/// <param name="PipelineId">The id of the pipeline that ran (or would have run).</param>
/// <param name="NodesBefore">Node count of each top-level graph before optimization, by graph name; empty when nothing was done.</param>
/// <param name="NodesAfter">Node count of each top-level graph after optimization, by graph name; empty when nothing was done.</param>
/// <param name="Statistics">One entry per pass run and graph, in execution order; empty when nothing was done.</param>
/// <param name="ArtifactSets">The number of artifact sets the written (or unchanged) package holds.</param>
/// <param name="PackageBytes">Size of the written (or unchanged) package in bytes.</param>
/// <param name="OptimizeTime">Wall time of the optimization phase.</param>
/// <param name="SaveTime">Wall time of writing the package.</param>
public sealed record PackageOptimizeResult(
    string PackagePath,
    bool Optimized,
    string PipelineId,
    IReadOnlyDictionary<string, int> NodesBefore,
    IReadOnlyDictionary<string, int> NodesAfter,
    IReadOnlyList<GraphPassStatistics> Statistics,
    int ArtifactSets,
    long PackageBytes,
    TimeSpan OptimizeTime,
    TimeSpan SaveTime);
