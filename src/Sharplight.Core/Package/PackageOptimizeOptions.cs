using Sharplight.Graph.Optimization;

namespace Sharplight.Package;

/// <summary>How <see cref="PackageOptimizer"/> optimizes a package.</summary>
public sealed class PackageOptimizeOptions
{
    /// <summary>The pipeline to run; <see langword="null"/> runs <see cref="GraphOptimizer.Default"/>.</summary>
    public GraphOptimizer? Optimizer { get; set; }

    /// <summary>Optimize even when the package already carries the pipeline's id (default false: such a package is left as it is).</summary>
    public bool Force { get; set; }

    /// <summary>
    /// Store the package's precompiled artifact sets with the result (default true). Code entries are content addressed (a kernel
    /// the optimized graphs no longer use is never asked for) and the driver validates pipeline caches, so they stay valid.
    /// </summary>
    public bool KeepArtifacts { get; set; } = true;

    /// <summary>Receives progress messages, or <see langword="null"/>.</summary>
    public Action<string>? Log { get; set; }
}
