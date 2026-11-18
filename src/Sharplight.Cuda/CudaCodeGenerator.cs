using System.Diagnostics;
using System.Globalization;
using Sharplight.Runtime;
using Sharplight.Shaders;
using Sharplight.Shaders.Lowering;

namespace Sharplight.Cuda;

/// <summary>
/// Everything CUDA code generation depends on (compute capability, shared memory, enabled features): lowers segments with the shared
/// <see cref="GraphLowerer"/> and compiles their Slang kernels to PTX (Slang to CUDA C++, NVRTC to PTX), and describes the resulting code
/// tier. Shared by the opened device (<see cref="CudaDevice"/>) and the compile-only device of a profile (<see cref="CudaProfileDevice"/>).
/// </summary>
internal sealed class CudaCodeGenerator
{
    /// <summary>Portable code format and artifact kind of the code tier (PTX per kernel).</summary>
    public const string CodeFormat = "ptx";

    /// <summary>Threads per warp, the subgroup size the kernels are specialized for.</summary>
    public const int WarpSize = 32;

    private const string CodeTargetPrefix = "ptx_";

    public CudaCodeGenerator(int computeCapability, long maxSharedMemory, IReadOnlySet<string> enabledFeatures, double submissionBudgetMs,
        int concurrentWorkgroups, int computeUnits)
    {
        ComputeCapability = computeCapability;
        MaxSharedMemory = maxSharedMemory;
        EnabledFeatures = enabledFeatures;
        SubmissionBudgetMs = submissionBudgetMs;
        ConcurrentWorkgroups = concurrentWorkgroups;
        ComputeUnits = computeUnits;
        Translator = new NvrtcTranslator(computeCapability);
        Compiler = new SlangCompiler(Translator);
    }

    /// <summary>The code target of a compute capability (<c>ptx_120</c> for 12.0): PTX for its virtual architecture.</summary>
    public static string CodeTargetOf(int computeCapability) => CodeTargetPrefix + computeCapability.ToString(CultureInfo.InvariantCulture);

    /// <summary>Parses a code target written by <see cref="CodeTargetOf"/>.</summary>
    public static bool TryParseCodeTarget(string? codeTarget, out int computeCapability)
    {
        computeCapability = 0;
        return codeTarget is not null && codeTarget.StartsWith(CodeTargetPrefix, StringComparison.Ordinal)
            && int.TryParse(codeTarget.AsSpan(CodeTargetPrefix.Length), NumberStyles.None, CultureInfo.InvariantCulture, out computeCapability);
    }

    public int ComputeCapability { get; }

    public string CodeTarget => CodeTargetOf(ComputeCapability);

    public long MaxSharedMemory { get; }

    public IReadOnlySet<string> EnabledFeatures { get; }

    public double SubmissionBudgetMs { get; }

    public int ConcurrentWorkgroups { get; }

    public int ComputeUnits { get; }

    public NvrtcTranslator Translator { get; }

    public SlangCompiler Compiler { get; }

    /// <summary>The code tier: PTX kernels (<c>kernel/*</c>), content addressed like the SPIR-V ones of the Vulkan provider.</summary>
    public ArtifactTarget CreateCodeTarget(string provider) => new(
        new Dictionary<string, string>
        {
            [ArtifactDescriptorKeys.Provider] = provider,
            [ArtifactDescriptorKeys.Tier] = ArtifactTiers.Code,
            [ArtifactDescriptorKeys.Kind] = CodeFormat,
            [ArtifactDescriptorKeys.Target] = CodeTarget,
            [ArtifactDescriptorKeys.Subgroup] = WarpSize.ToString(CultureInfo.InvariantCulture),
            [ArtifactDescriptorKeys.SharedMemory] = MaxSharedMemory.ToString(CultureInfo.InvariantCulture),
            [ArtifactDescriptorKeys.Features] = string.Join(",", EnabledFeatures.Order(StringComparer.Ordinal)),
            [ArtifactDescriptorKeys.Compiler] = SlangCompiler.CompilerId + "+" + Translator.Id,
        },
        [ArtifactDescriptorKeys.Target],
        [SlangCompiler.KernelKeyPrefix]);

    /// <summary>Lowers <paramref name="segment"/>; its kernels may still be compiling when this returns.</summary>
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
            Target = ShaderTarget.Ptx,
            SubgroupSize = WarpSize,
            MaxSharedMemory = MaxSharedMemory,
            Features = options.FeaturesFor(device),
            Compile = options,
            SubmissionBudgetMs = SubmissionBudgetMs,
            ConcurrentWorkgroups = ConcurrentWorkgroups,
            ComputeUnits = ComputeUnits,
            Cache = cache,
        });
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
        SubgroupSize = WarpSize,
        MaxSharedMemory = MaxSharedMemory,
        Features = [.. EnabledFeatures.Order(StringComparer.Ordinal)],
        SubmissionBudgetMs = SubmissionBudgetMs,
        ConcurrentWorkgroups = ConcurrentWorkgroups,
        Properties = properties,
    };

    /// <summary>The submission budget of a device: <paramref name="detected"/> unless the device setting or the environment overrides it.</summary>
    public static double EffectiveSubmissionBudget(double detected, DeviceOptions options)
    {
        options.Settings.TryGetValue("submission_budget_ms", out var setting);
        return DiagnosticSwitches.SubmissionBudgetMs(setting) is { } budgetMs ? Math.Max(0, budgetMs) : detected;
    }

    /// <summary>The concurrent-workgroup hint of a device: <paramref name="detected"/> unless the device setting or the environment overrides it.</summary>
    public static int EffectiveConcurrentWorkgroups(int detected, DeviceOptions options)
    {
        options.Settings.TryGetValue("concurrent_workgroups", out var setting);
        return DiagnosticSwitches.ConcurrentWorkgroups(setting) is { } workgroups ? Math.Max(0, workgroups) : detected;
    }

    /// <summary>The compute units (SMs) a device or profile reports in its capabilities, or 0.</summary>
    public static int ComputeUnitsOf(IReadOnlyDictionary<string, string> capabilities)
        => capabilities.TryGetValue("compute_units", out var text) && int.TryParse(text, CultureInfo.InvariantCulture, out int units) ? units : 0;
}
