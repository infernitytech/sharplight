using System.Diagnostics;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Shaders;

namespace Sharplight.Vulkan;

/// <summary>
/// A compile-only Vulkan device for a <see cref="DeviceProfile"/> (see <see cref="IOfflineCompilationProvider.OpenProfile"/>):
/// lowers segments and compiles their kernels to SPIR-V exactly like a <see cref="VulkanDevice"/> with that profile, without a
/// Vulkan driver. Only the code tier is produced; the compiled segments cannot run.
/// </summary>
internal sealed class VulkanProfileDevice : IComputeDevice
{
    /// <summary>Memory assumed when the profile does not state it, so placement never runs out.</summary>
    private const long UnknownMemoryBytes = 1L << 40;

    private readonly SpirvCodeGenerator _code;

    public VulkanProfileDevice(DeviceProfile profile, DeviceOptions options)
    {
        Profile = profile;
        Info = new DeviceInfo
        {
            Provider = profile.Provider,
            Index = -1,
            Name = profile.Name,
            Kind = profile.Kind,
            TotalMemory = profile.TotalMemory > 0 ? profile.TotalMemory : UnknownMemoryBytes,
            Vendor = profile.Vendor,
            Features = new HashSet<string>(profile.Features, StringComparer.OrdinalIgnoreCase),
            SubmissionBudgetMs = profile.SubmissionBudgetMs,
            Capabilities = profile.Properties,
        };
        EnabledFeatures = DeviceFeatures.Effective(Info.Features, options.DisabledFeatures);
        _code = new SpirvCodeGenerator(profile.SubgroupSize, profile.MaxSharedMemory, EnabledFeatures,
            SpirvCodeGenerator.EffectiveSubmissionBudget(profile.SubmissionBudgetMs, options),
            options.KernelCacheDirectory ?? SlangCompiler.DefaultCacheDirectory);
        ArtifactTargets = [_code.CreateCodeTarget(Info.Provider)];
    }

    /// <summary>The profile the device was opened for.</summary>
    public DeviceProfile Profile { get; }

    public DeviceInfo Info { get; }

    public IReadOnlySet<string> EnabledFeatures { get; }

    /// <summary>Only the code tier: binaries need the driver.</summary>
    public IReadOnlyList<ArtifactTarget> ArtifactTargets { get; }

    public long AvailableMemory => Info.TotalMemory;

    public OpSupport Supports(Node node) => Shaders.Lowering.GraphLowerer.Supports(node);

    /// <summary>Lowers the segment and waits for all its kernels; a kernel Slang rejects fails the compilation.</summary>
    /// <exception cref="KernelCompilationException">Slang rejected one of the kernels.</exception>
    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options)
    {
        var cache = options.CacheFor?.Invoke(this);
        var program = _code.Lower(segment, this, options, cache, out var kernels);
        var stopwatch = Stopwatch.StartNew();
        try
        {
            Task.WaitAll([.. kernels.Select(k => k.CodeTask)]);
        }
        catch (AggregateException e)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerExceptions[0]).Throw();
        }

        if (DiagnosticSwitches.LoadTiming)
        {
            Console.Error.WriteLine($"[load] kernels segment {segment.Index}: waited {stopwatch.ElapsedMilliseconds} ms for {kernels.Count} kernels; slang {SlangCompiler.Statistics}");
        }

        return new CompileOnlySegment(segment, program.Launches.Count);
    }

    public DeviceProfile CreateProfile() => Profile;

    /// <summary>Nothing to release: the device holds no driver objects.</summary>
    public void Dispose()
    {
    }
}
