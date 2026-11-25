using System.Diagnostics;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Shaders;
using Sharplight.Shaders.Lowering;

namespace Sharplight.Cuda;

/// <summary>
/// A compile-only CUDA device for a <see cref="DeviceProfile"/>, the device of the portable stage (<see cref="CudaProvider.CompilePortableAsync"/>):
/// lowers segments and compiles their kernels to PTX exactly like a <see cref="CudaDevice"/> with that profile, without a GPU or driver
/// (NVRTC is needed). Only the code tier is produced; the compiled segments cannot run. Memory figures come from
/// <see cref="DeviceProfile.TotalMemory"/> (plenty when unknown) so placement works as usual.
/// </summary>
internal sealed class CudaProfileDevice : IComputeDevice
{
    /// <summary>Memory assumed when the profile does not state it, so placement never runs out.</summary>
    private const long UnknownMemoryBytes = 1L << 40;

    private readonly CudaCodeGenerator _code;
    /// <summary>Records compilations that come without a model-level cache (graphs built in code) into the artifact store.</summary>
    private readonly CompilationArtifacts _ownArtifacts = new();

    /// <exception cref="ArgumentException">The profile's code target is not <c>ptx_XY</c>.</exception>
    public CudaProfileDevice(DeviceProfile profile, DeviceOptions options)
    {
        if (!CudaCodeGenerator.TryParseCodeTarget(profile.CodeTarget, out int computeCapability))
        {
            throw new ArgumentException($"Code target '{profile.CodeTarget}' is not a CUDA target (ptx_XY).", nameof(profile));
        }

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
            ConcurrentWorkgroups = profile.ConcurrentWorkgroups,
            Capabilities = profile.Properties,
        };
        EnabledFeatures = DeviceFeatures.Effective(Info.Features, options.DisabledFeatures);
        _code = new CudaCodeGenerator(computeCapability, profile.MaxSharedMemory, EnabledFeatures,
            options.ResolveSubmissionBudgetMs(profile.SubmissionBudgetMs),
            options.ResolveConcurrentWorkgroups(profile.ConcurrentWorkgroups), CudaCodeGenerator.ComputeUnitsOf(profile.Properties));
        ArtifactStore = ArtifactStore.For(options);
        ArtifactTargets = [_code.CreateCodeTarget(Info.Provider)];
    }

    /// <summary>The persistent artifact store this device's compilations consult and record into.</summary>
    public ArtifactStore? ArtifactStore { get; }

    /// <summary>The profile the device was opened for.</summary>
    public DeviceProfile Profile { get; }

    public DeviceInfo Info { get; }

    public IReadOnlySet<string> EnabledFeatures { get; }

    /// <summary>Only the code tier: cubins need the driver.</summary>
    public IReadOnlyList<ArtifactTarget> ArtifactTargets { get; }

    public long AvailableMemory => Info.TotalMemory;

    public OpSupport Supports(Node node) => GraphLowerer.Supports(node);

    /// <summary>Lowers the segment and waits for all its kernels; a kernel Slang or NVRTC rejects fails the compilation.</summary>
    /// <exception cref="KernelCompilationException">Slang or NVRTC rejected one of the kernels.</exception>
    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options)
    {
        var cache = options.CacheFor?.Invoke(this) ?? _ownArtifacts.For(this);
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
