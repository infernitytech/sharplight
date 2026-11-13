using System.Diagnostics;
using System.Globalization;
using Sharplight.Runtime;
using Sharplight.Shaders;
using Sharplight.Shaders.Lowering;

namespace Sharplight.Vulkan;

/// <summary>
/// Everything Vulkan code generation depends on (subgroup size, shared memory, enabled features, submission budget): lowers
/// segments and compiles their kernels to SPIR-V, and describes the resulting code tier. Shared by the opened device
/// (<see cref="VulkanDevice"/>) and the compile-only device of a profile (<see cref="VulkanProfileDevice"/>), so both produce the
/// same kernels, cache keys and code-tier descriptor for the same profile.
/// </summary>
internal sealed class SpirvCodeGenerator
{
    /// <summary>Artifact kind of the code tier.</summary>
    public const string CodeKind = "spirv";

    public SpirvCodeGenerator(int subgroupSize, long maxSharedMemory, IReadOnlySet<string> enabledFeatures, double submissionBudgetMs, string kernelCacheDirectory)
    {
        SubgroupSize = subgroupSize;
        MaxSharedMemory = maxSharedMemory;
        EnabledFeatures = enabledFeatures;
        SubmissionBudgetMs = submissionBudgetMs;
        Compiler = new SlangCompiler(kernelCacheDirectory);
    }

    /// <summary>The Slang profile of every kernel (<c>spirv_1_6</c>).</summary>
    public static string CodeTarget => SlangCompiler.TargetProfile(ShaderTarget.SpirV);

    public int SubgroupSize { get; }

    public long MaxSharedMemory { get; }

    /// <summary>Detected features minus the device-level disables.</summary>
    public IReadOnlySet<string> EnabledFeatures { get; }

    /// <summary>Effective submission budget in ms (0 = unlimited).</summary>
    public double SubmissionBudgetMs { get; }

    public SlangCompiler Compiler { get; }

    /// <summary>
    /// The code tier: SPIR-V kernels (<c>kernel/*</c>), content addressed, so any stored set for the same SPIR-V target may be
    /// consulted; the other descriptor keys record what the set was generated for.
    /// </summary>
    public ArtifactTarget CreateCodeTarget(string provider) => new(
        new Dictionary<string, string>
        {
            [ArtifactDescriptorKeys.Provider] = provider,
            [ArtifactDescriptorKeys.Tier] = ArtifactTiers.Code,
            [ArtifactDescriptorKeys.Kind] = CodeKind,
            [ArtifactDescriptorKeys.Target] = CodeTarget,
            [ArtifactDescriptorKeys.Subgroup] = SubgroupSize.ToString(CultureInfo.InvariantCulture),
            [ArtifactDescriptorKeys.SharedMemory] = MaxSharedMemory.ToString(CultureInfo.InvariantCulture),
            [ArtifactDescriptorKeys.Features] = string.Join(",", EnabledFeatures.Order(StringComparer.Ordinal)),
            [ArtifactDescriptorKeys.Compiler] = SlangCompiler.CompilerId,
        },
        [ArtifactDescriptorKeys.Target],
        [SlangCompiler.KernelKeyPrefix]);

    /// <summary>
    /// Lowers <paramref name="segment"/> for <paramref name="device"/>; its kernels may still be compiling when this returns
    /// (see <see cref="CompiledKernel.Code"/>).
    /// </summary>
    /// <param name="segment">The segment.</param>
    /// <param name="device">The device compiling (for the per-compilation feature set).</param>
    /// <param name="options">Model-level compile options.</param>
    /// <param name="cache">Model-level compilation cache, or <see langword="null"/>.</param>
    /// <param name="kernels">The distinct kernels of the program, in launch order.</param>
    public KernelProgram Lower(GraphSegment segment, IComputeDevice device, CompileOptions options, ICompilationCache? cache, out IReadOnlyList<CompiledKernel> kernels)
    {
        var stopwatch = Stopwatch.StartNew();
        var program = GraphLowerer.Lower(segment, Compiler, new LoweringOptions
        {
            Target = ShaderTarget.SpirV,
            SubgroupSize = SubgroupSize,
            MaxSharedMemory = MaxSharedMemory,
            Features = options.FeaturesFor(device),
            Compile = options,
            SubmissionBudgetMs = SubmissionBudgetMs,
            Cache = cache,
        });
        // Variants of different launches often share one kernel.
        kernels = [.. program.Launches.Concat(program.ResetLaunches).SelectMany(l => l.Variants).Select(v => v.Kernel).DistinctBy(k => k.CacheKey)];
        if (DiagnosticSwitches.LoadTiming)
        {
            Console.Error.WriteLine($"[load] lower segment {segment.Index}: {stopwatch.ElapsedMilliseconds} ms ({program.Launches.Count} launches, {program.Buffers.Count} buffers)");
        }

        return program;
    }

    /// <summary>The profile of a device with these code generation settings.</summary>
    public DeviceProfile CreateProfile(DeviceInfo info, IReadOnlyDictionary<string, string> properties) => new()
    {
        Provider = info.Provider,
        Name = info.Name,
        Vendor = info.Vendor,
        Kind = info.Kind,
        TotalMemory = info.TotalMemory,
        CodeTarget = CodeTarget,
        SubgroupSize = SubgroupSize,
        MaxSharedMemory = MaxSharedMemory,
        Features = [.. EnabledFeatures.Order(StringComparer.Ordinal)],
        SubmissionBudgetMs = SubmissionBudgetMs,
        Properties = properties,
    };

    /// <summary>
    /// The submission budget of a device: <paramref name="detected"/> unless the device setting <c>submission_budget_ms</c> or the
    /// environment overrides it.
    /// </summary>
    public static double EffectiveSubmissionBudget(double detected, DeviceOptions options)
    {
        options.Settings.TryGetValue("submission_budget_ms", out var setting);
        return DiagnosticSwitches.SubmissionBudgetMs(setting) is { } budgetMs ? Math.Max(0, budgetMs) : detected;
    }
}
