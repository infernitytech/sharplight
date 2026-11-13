using Sharplight.Runtime;

namespace Sharplight.Shaders.Lowering;


public sealed class LoweringOptions
{
    public ShaderTarget Target { get; init; } = ShaderTarget.SpirV;
    public int SubgroupSize { get; init; } = 32;
    /// <summary>Shared (workgroup) memory per workgroup in bytes.</summary>
    public long MaxSharedMemory { get; init; } = 32768;
    /// <summary>
    /// Effective device features (<see cref="DeviceFeatures"/>) the generated code may use. Every
    /// feature-specific kernel has a portable fallback used when its feature is absent.
    /// </summary>
    public IReadOnlySet<string> Features { get; init; } = new HashSet<string>();
    public bool Has(string feature) => Features.Contains(feature);
    /// <summary>See <see cref="DeviceInfo.SubmissionBudgetMs"/>: when > 0, single dispatches whose work grows with the batch (attention) are sliced to about this duration.</summary>
    public double SubmissionBudgetMs { get; init; }
    public required CompileOptions Compile { get; init; }
    /// <summary>Model-level compilation cache for this device (see <see cref="CompileOptions.CacheFor"/>).</summary>
    public Runtime.ICompilationCache? Cache { get; init; }
}
