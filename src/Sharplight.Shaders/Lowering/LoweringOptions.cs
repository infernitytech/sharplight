using Sharplight.Runtime;

namespace Sharplight.Shaders.Lowering;

/// <summary>Target and device capabilities that <see cref="GraphLowerer"/> specializes kernels for.</summary>
public sealed class LoweringOptions
{
    /// <summary>Code generation target.</summary>
    public ShaderTarget Target { get; init; } = ShaderTarget.SpirV;
    /// <summary>Lanes per subgroup of the device.</summary>
    public int SubgroupSize { get; init; } = 32;
    /// <summary>Shared (workgroup) memory per workgroup in bytes.</summary>
    public long MaxSharedMemory { get; init; } = 32768;
    /// <summary>
    /// Effective device features (<see cref="DeviceFeatures"/>) the generated code may use. Every
    /// feature-specific kernel has a portable fallback used when its feature is absent.
    /// </summary>
    public IReadOnlySet<string> Features { get; init; } = new HashSet<string>();
    /// <summary>True when <paramref name="feature"/> (a <see cref="DeviceFeatures"/> name) is available.</summary>
    public bool Has(string feature) => Features.Contains(feature);
    /// <summary>See <see cref="DeviceInfo.SubmissionBudgetMs"/>: when > 0, single dispatches whose work grows with the batch (attention) are sliced to about this duration.</summary>
    public double SubmissionBudgetMs { get; init; }
    /// <summary>See <see cref="DeviceInfo.ConcurrentWorkgroups"/>: workgroups the device runs at once, 0 = unknown (a large GPU).</summary>
    public int ConcurrentWorkgroups { get; init; }
    /// <summary>
    /// Compute units of the device (NVIDIA SMs, AMD CUs), 0 = unknown. Sizes the K splits of the tensor-core GEMMs: a device with
    /// fewer units fills its waves with fewer tiles, and a split it does not need only adds partial sums and a reduction.
    /// </summary>
    public int ComputeUnits { get; init; }
    /// <summary>
    /// True (default) when workgroups of one dispatch can hand results to each other through device-scope fences and atomics (a
    /// workgroup publishes partial results, the last one to arrive acquires and merges them). False on Metal, whose memory model
    /// only has relaxed atomics and threadgroup-scope barriers within a dispatch: kernels then merge in a separate launch.
    /// </summary>
    public bool CrossWorkgroupFences { get; init; } = true;
    /// <summary>Model-level compile options (symbol bounds, context length, precision policy).</summary>
    public required CompileOptions Compile { get; init; }
    /// <summary>Model-level compilation cache for this device (see <see cref="CompileOptions.CacheFor"/>).</summary>
    public Runtime.ICompilationCache? Cache { get; init; }
}
