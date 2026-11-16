using System.Diagnostics;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Package;

/// <summary>
/// One run of <see cref="PackagePrecompiler"/>: keeps the package open (its stored sets are reused and compared), runs the two stages
/// per device with sharing between identical devices, and writes every new set in one append at the end.
/// </summary>
internal sealed class PrecompileSession : IDisposable
{
    private const string TemporaryCachePrefix = "sharplight-precompile-";

    private readonly string _path;
    private readonly PrecompileOptions _options;
    private readonly IProgress<PrecompileProgress>? _progress;
    private readonly CancellationToken _cancellationToken;
    private readonly ModelSource _source;
    private readonly string? _temporaryCache;
    private readonly Stopwatch _stopwatch = Stopwatch.StartNew();
    /// <summary>Portable sets by code-tier descriptor id, with the device they were produced for.</summary>
    private readonly Dictionary<string, (PortableCodeSet Set, string Device)> _portable = new(StringComparer.Ordinal);
    /// <summary>Device sets by device code identity (<see cref="CompilationArtifacts.DeviceCodeIdentity"/>), with the device they were produced on.</summary>
    private readonly Dictionary<string, (DeviceCodeSet Set, string Device)> _deviceCode = new(StringComparer.Ordinal);
    private readonly List<DevicePrecompileResult> _results = [];
    private readonly List<SkippedProvider> _skipped = [];

    public PrecompileSession(string path, PrecompileOptions options, IProgress<PrecompileProgress>? progress, CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        _path = path;
        _options = options;
        _progress = progress;
        _cancellationToken = cancellationToken;
        _source = new ModelPackageImporter().Open(path);
        // A fresh kernel cache directory: every kernel is generated from scratch (reproducible), unless the user cache is wanted.
        _temporaryCache = options.UseUserKernelCache ? null : Directory.CreateTempSubdirectory(TemporaryCachePrefix).FullName;
    }

    private bool StoresCode => _options.Tiers is null || _options.Tiers.Contains(ArtifactTiers.Code);

    private bool WantsDeviceCode => _options.Tiers is null || _options.Tiers.Contains(ArtifactTiers.Binary);

    /// <summary>Every device of <paramref name="provider"/>; an unavailable provider is reported as skipped.</summary>
    public async Task CompileProviderAsync(IExecutionProvider provider)
    {
        var availability = Safe(provider.CheckAvailability, ProviderAvailability.Unavailable);
        if (!availability.IsAvailable)
        {
            Skip(provider.Name, "", availability.Reason.Length > 0 ? availability.Reason : "not available");
            return;
        }

        var devices = Safe(provider.GetDevices, _ => []);
        if (devices.Count == 0)
        {
            Skip(provider.Name, "", "no devices");
            return;
        }

        foreach (var device in devices)
        {
            await CompileDeviceAsync(provider, device).ConfigureAwait(false);
        }
    }

    /// <summary>Both stages for one device; a device that fails (driver error, ...) is reported as skipped.</summary>
    public async Task CompileDeviceAsync(IExecutionProvider provider, DeviceInfo info)
    {
        _cancellationToken.ThrowIfCancellationRequested();
        Report(provider.Name, info.Name, PrecompileStage.Opening, $"Opening {info}");
        IComputeDevice device;
        try
        {
            device = provider.OpenDevice(info, DeviceOptions(includeCache: false));
        }
        catch (Exception error) when (IsDeviceFailure(error))
        {
            Skip(provider.Name, info.Name, $"cannot open the device: {error.Message}");
            return;
        }

        using (device)
        {
            try
            {
                await CompileOpenedDeviceAsync(provider, device).ConfigureAwait(false);
            }
            catch (Exception error) when (IsDeviceFailure(error))
            {
                Skip(provider.Name, info.Name, $"failed: {error.Message}");
            }
        }
    }

    /// <summary>The portable stage for a profile (no device).</summary>
    public async Task CompileProfileAsync(IExecutionProvider provider, DeviceProfile profile)
    {
        var (portable, elapsed) = await CompilePortableAsync(provider, profile, profile.Name).ConfigureAwait(false);
        _portable[portable.Id] = (portable, profile.Name);
        _results.Add(new DevicePrecompileResult(provider.Name, profile.Name, CodeSetSummary.Of(portable, null, elapsed), null,
            portable.IsEmpty ? "nothing to precompile" : "device code needs the device"));
        Report(provider.Name, profile.Name, PrecompileStage.Completed, $"{portable.Entries.Count} kernels");
    }

    /// <summary>Writes the new sets into the package (one crash-safe append) and builds the report.</summary>
    public PackagePrecompileReport Store()
    {
        _cancellationToken.ThrowIfCancellationRequested();
        var stored = _source.CompilationCaches;
        var produced = new List<CodeSet>();
        if (StoresCode)
        {
            produced.AddRange(_portable.Values.Select(p => p.Set).Where(s => !s.IsEmpty));
        }

        produced.AddRange(_deviceCode.Values.Select(d => d.Set).Where(s => !s.IsEmpty));
        var write = produced.Where(set => !IsStored(set, stored)).ToList();
        var unchanged = produced.Except(write).Select(set => ArtifactDescriptor.Format(set.Descriptor)).ToList();
        foreach (var set in unchanged)
        {
            _options.Log?.Invoke($"Already stored: {set}");
        }

        if (write.Count > 0)
        {
            Report("", "", PrecompileStage.Writing, $"Writing {write.Count} artifact set(s) into {_path}");
            ModelPackage.AddArtifacts(_path, write.Select(set => set.ToArtifactSet()));
        }

        foreach (var set in write)
        {
            _options.Log?.Invoke($"Stored {set}");
        }

        return new PackagePrecompileReport(_path, [.. _results], [.. _skipped], [.. write.Select(s => ArtifactDescriptor.Format(s.Descriptor))], unchanged,
            write.Sum(s => s.TotalBytes), new FileInfo(_path).Length, _stopwatch.Elapsed);
    }

    public void Dispose()
    {
        foreach (var resource in _source.OwnedResources)
        {
            resource.Dispose();
        }

        if (_temporaryCache is not null)
        {
            try
            {
                Directory.Delete(_temporaryCache, recursive: true);
            }
            catch (IOException)
            {
                // Best effort: a leftover temporary kernel cache only costs disk space.
            }
        }
    }

    private async Task CompileOpenedDeviceAsync(IExecutionProvider provider, IComputeDevice device)
    {
        string name = device.Info.Name;
        var codeTarget = device.ArtifactTargets.FirstOrDefault(t => t.Tier == ArtifactTiers.Code);
        var profile = device.CreateProfile();
        PortableCodeSet portable;
        CodeSetSummary? portableSummary;
        if (codeTarget is not null && _portable.TryGetValue(codeTarget.Id, out var shared))
        {
            portable = shared.Set;
            portableSummary = CodeSetSummary.Of(portable, shared.Device, TimeSpan.Zero);
        }
        else
        {
            var (compiled, elapsed) = await CompilePortableAsync(provider, profile, name).ConfigureAwait(false);
            portable = compiled;
            portableSummary = CodeSetSummary.Of(portable, null, elapsed);
            if (!portable.IsEmpty)
            {
                _portable[portable.Id] = (portable, name);
            }
        }

        var identity = CompilationArtifacts.DeviceCodeIdentity(device);
        CodeSetSummary? deviceSummary = null;
        if (!WantsDeviceCode)
        {
            deviceSummary = null;
        }
        else if (identity.Length > 0 && _deviceCode.TryGetValue(identity, out var sharedCode))
        {
            deviceSummary = CodeSetSummary.Of(sharedCode.Set, sharedCode.Device, TimeSpan.Zero);
        }
        else
        {
            Report(provider.Name, name, PrecompileStage.DeviceSpecific, $"Compiling device code for {name}");
            var stopwatch = Stopwatch.StartNew();
            var deviceCode = await provider.CompileDeviceSpecificAsync(device, portable, _cancellationToken).ConfigureAwait(false);
            deviceSummary = CodeSetSummary.Of(deviceCode, null, stopwatch.Elapsed);
            if (identity.Length > 0 && !deviceCode.IsEmpty)
            {
                _deviceCode[identity] = (deviceCode, name);
            }
        }

        bool nothing = portable.IsEmpty && (deviceSummary is null || deviceSummary.Entries == 0);
        string? note = nothing ? (provider.PortableCodeFormat is null ? "nothing to precompile (kernels are built into the provider)" : "nothing to precompile") : null;
        _results.Add(new DevicePrecompileResult(provider.Name, name, portable.IsEmpty ? null : portableSummary,
            deviceSummary is null || deviceSummary.Entries == 0 ? null : deviceSummary, note));
        Report(provider.Name, name, nothing ? PrecompileStage.Skipped : PrecompileStage.Completed, note ?? Summary(portableSummary, deviceSummary));
    }

    private async Task<(PortableCodeSet Set, TimeSpan Elapsed)> CompilePortableAsync(IExecutionProvider provider, DeviceProfile? profile, string device)
    {
        Report(provider.Name, device, PrecompileStage.Portable, $"Generating {provider.PortableCodeFormat ?? "portable code"} for {device}");
        var kernels = new ForwardingProgress(kernelProgress =>
        {
            _options.Progress?.Report(kernelProgress);
            Report(provider.Name, device, PrecompileStage.Portable, kernelProgress.Stage.ToString(), kernelProgress);
        });
        var compilation = new ModelCompilation(_path, _source.Definition!, _options.WithProgress(kernels));
        var request = new PortableCompilationRequest(profile, compilation.Workload)
        {
            Stored = _source.CompilationCaches,
            DeviceOptions = DeviceOptions(includeCache: true),
        };
        var stopwatch = Stopwatch.StartNew();
        var set = await provider.CompilePortableAsync(request, _cancellationToken).ConfigureAwait(false);
        return (set, stopwatch.Elapsed);
    }

    /// <summary>
    /// Whether the package already holds <paramref name="set"/>: everything was served from the stored sets, or a stored set with the same
    /// descriptor has every entry (portable code is content addressed; device code is regenerated only when entries are missing).
    /// </summary>
    private static bool IsStored(CodeSet set, ICompilationCacheStore? stored)
    {
        if (set is PortableCodeSet && set.ReusedEntries == set.Entries.Count)
        {
            return true;
        }

        var match = stored?.ArtifactSets.FirstOrDefault(s => ArtifactDescriptor.Id(s.Descriptor) == set.Id);
        return match is not null && set.Entries.Keys.All(key => match.Entries.TryGet(key, out _));
    }

    private DeviceOptions DeviceOptions(bool includeCache)
    {
        var options = new DeviceOptions { KernelCacheDirectory = includeCache ? _temporaryCache : null };
        foreach (var feature in _options.DisabledFeatures)
        {
            options.DisabledFeatures.Add(feature);
        }

        return options;
    }

    private void Skip(string provider, string device, string reason)
    {
        _skipped.Add(new SkippedProvider(provider, device, reason));
        _options.Log?.Invoke(device.Length == 0 ? $"Skipped {provider}: {reason}" : $"Skipped {provider} {device}: {reason}");
        Report(provider, device, PrecompileStage.Skipped, reason);
    }

    private void Report(string provider, string device, PrecompileStage stage, string message, LoadProgress? kernels = null)
        => _progress?.Report(new PrecompileProgress(provider, device, stage, message, kernels));

    private static string Summary(CodeSetSummary? portable, CodeSetSummary? device)
    {
        var parts = new List<string>();
        if (portable is not null && portable.Entries > 0)
        {
            parts.Add($"{portable.Entries} {portable.Kind} kernels{(portable.SharedWith is null ? "" : $" (shared with {portable.SharedWith})")}");
        }

        if (device is not null && device.Entries > 0)
        {
            parts.Add($"{device.Kind} {device.Entries} entries{(device.SharedWith is null ? "" : $" (shared with {device.SharedWith})")}");
        }

        return string.Join(", ", parts);
    }

    /// <summary>Failures that only affect one device or provider (driver errors), as opposed to model errors and cancellation.</summary>
    private static bool IsDeviceFailure(Exception error)
        => error is not (OperationCanceledException or MissingSymbolBoundException or OutOfMemoryException);

    private static T Safe<T>(Func<T> query, Func<string, T> onError)
    {
        try
        {
            return query();
        }
        catch (Exception error) when (IsDeviceFailure(error))
        {
            return onError(error.Message);
        }
    }
}
