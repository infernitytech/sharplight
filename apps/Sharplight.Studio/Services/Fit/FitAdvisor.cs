using System.Globalization;
using Sharplight.Hosting.Fit;
using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Services.Fit;

/// <summary>
/// Fit estimates for the views: builds <see cref="FitQuery"/>s from the runtime's devices, resident models and the residency
/// settings, plans them with <see cref="FitPlanner"/>, and finds changes that make a model fit (shorter context, smaller KV cache,
/// fewer slots, CPU offload, splitting over the GPUs, a smaller quantization in the library).
/// </summary>
internal sealed class FitAdvisor
{
    private const int MinSuggestedContext = 1024;
    private const int ContextSteps = 5;

    private readonly IModelRuntime _runtime;
    private readonly ISettingsService _settings;
    private readonly DeviceMemoryWatch? _memory;

    /// <param name="runtime">The devices, resident models and footprint estimates.</param>
    /// <param name="settings">The residency reserve and the default load settings.</param>
    /// <param name="memory">The shared memory figures, refreshed by every read here; null reads the runtime directly.</param>
    public FitAdvisor(IModelRuntime runtime, ISettingsService settings, DeviceMemoryWatch? memory = null)
    {
        _runtime = runtime;
        _settings = settings;
        _memory = memory;
    }

    /// <summary>The advisor of <paramref name="services"/>, reading memory through their shared <see cref="DeviceMemoryWatch"/>.</summary>
    /// <param name="services">The services.</param>
    public static FitAdvisor For(StudioServices services) => new(services.Runtime, services.Settings, services.Memory);

    /// <summary>Memory kept free on every device, in bytes.</summary>
    public long ReserveBytes => Math.Max(0, _settings.Current.Residency.ReserveMiB) * (1L << 20);

    /// <summary>The device a load with <paramref name="deviceId"/> goes to (the runtime's default when it names none it knows).</summary>
    /// <param name="deviceId">The device id of the load settings.</param>
    public ComputeDeviceOption? ResolveDevice(string? deviceId)
        => _runtime.Devices.FirstOrDefault(device => device.Id == deviceId) ?? _runtime.Devices.FirstOrDefault();

    /// <summary>The memory of every physical device now.</summary>
    /// <param name="cancellationToken">Cancels the query.</param>
    public Task<IReadOnlyList<DeviceMemory>> GetMemoryAsync(CancellationToken cancellationToken)
        => _memory?.RefreshAsync(cancellationToken) ?? _runtime.GetDeviceMemoryAsync(_settings.Current.DisabledFeatures, cancellationToken);

    /// <summary>The query for loading a model of <paramref name="footprint"/> with <paramref name="settings"/>.</summary>
    /// <param name="footprint">The estimated footprint.</param>
    /// <param name="settings">The load settings (device and CPU offload).</param>
    /// <param name="memory">The devices' memory.</param>
    public FitQuery Query(ModelFootprint footprint, LoadSettings settings, IReadOnlyList<DeviceMemory> memory)
    {
        var device = ResolveDevice(settings.DeviceId);
        var gpus = memory.Where(entry => entry.IsGpu).ToList();
        IReadOnlyList<DeviceMemory> targets = device is null ? []
            : device.IsSplit ? gpus
            : [.. memory.Where(entry => entry.Id == device.Id)];
        // Offloaded layers go to the runtime's CPU device (the first non-GPU one).
        var cpu = _runtime.Devices.FirstOrDefault(option => !option.IsGpu);
        var system = memory.FirstOrDefault(entry => entry.Id == cpu?.Id) ?? memory.FirstOrDefault(entry => !entry.IsGpu);
        return new FitQuery
        {
            Footprint = footprint,
            Targets = targets,
            SystemMemory = system,
            ReserveBytes = ReserveBytes,
            Residents = [.. _runtime.LoadedModels.Select(model => Usage(model, gpus))],
            CpuOffload = settings.CpuOffload && device?.IsGpu == true && system is not null,
            ExpertsOnDisk = settings.ExpertsOnDisk && settings.CpuOffload,
        };
    }

    /// <summary>Plans loading a model of <paramref name="footprint"/> with <paramref name="settings"/>.</summary>
    /// <param name="footprint">The estimated footprint.</param>
    /// <param name="settings">The load settings.</param>
    /// <param name="memory">The devices' memory.</param>
    public FitReport Plan(ModelFootprint footprint, LoadSettings settings, IReadOnlyList<DeviceMemory> memory)
        => FitPlanner.Plan(Query(footprint, settings, memory));

    /// <summary>
    /// Plans a model that is not downloaded yet for the default load settings (with CPU offload, to tell how much would run on the CPU):
    /// on the default device, or split over every GPU when that fits better.
    /// </summary>
    /// <param name="footprint">The approximate footprint (<see cref="DownloadEstimate"/>).</param>
    /// <param name="memory">The devices' memory.</param>
    /// <returns>The report and whether it is for the split over every GPU.</returns>
    public (FitReport Report, bool Split) PlanDownload(ModelFootprint footprint, IReadOnlyList<DeviceMemory> memory)
    {
        var settings = _settings.Current.DefaultLoad with { CpuOffload = true };
        var report = Plan(footprint, settings, memory);
        var device = ResolveDevice(settings.DeviceId);
        if (report.Verdict == FitVerdict.Fits || device?.IsSplit == true || !_runtime.Devices.Any(option => option.IsSplit))
        {
            return (report, device?.IsSplit == true);
        }

        var split = Plan(footprint, settings with { DeviceId = ComputeDeviceOption.AllGpusId }, memory);
        return split.Verdict < report.Verdict ? (split, true) : (report, false);
    }

    /// <summary>
    /// Changes to <paramref name="request"/> that make the model fit, each with the verdict it leads to: for every kind of change the
    /// smallest one that fits, else the one that improves the most. Empty when it fits as configured.
    /// </summary>
    /// <param name="request">The load as configured.</param>
    /// <param name="report">Its plan.</param>
    /// <param name="memory">The devices' memory the plan used.</param>
    /// <param name="entry">The library model being loaded, or null.</param>
    /// <param name="library">The library, for smaller quantizations of the same model.</param>
    /// <param name="cancellationToken">Cancels the estimates.</param>
    /// <param name="contexts">
    /// The longest contexts that fit (<see cref="SizeContextAsync"/>), when the caller shows them (<see cref="SizingAsync"/>): then no shorter
    /// context is suggested here. Null suggests the first of a few halvings of the context that fits.
    /// </param>
    /// <param name="expertBytes">The routed experts' bytes, 0 for a dense model: a mixture-of-experts model that does not fit memory is offered experts on disk.</param>
    public async Task<IReadOnlyList<FitSuggestion>> SuggestAsync(ModelLoadRequest request, FitReport report, IReadOnlyList<DeviceMemory> memory,
        LibraryEntry? entry, IReadOnlyList<LibraryEntry> library, CancellationToken cancellationToken, ContextFit? contexts = null, long expertBytes = 0)
    {
        if (report.FitsOnDevice)
        {
            return [];
        }

        var settings = request.Settings;
        var suggestions = new List<FitSuggestion>();

        if (report.Verdict == FitVerdict.NeedsOffload && !report.CpuOffload)
        {
            suggestions.Add(new FitSuggestion($"Turn on CPU offload · about {Percent(report.CpuShare)} runs on the CPU", FitVerdict.NeedsOffload,
                settings with { CpuOffload = true }));
        }

        if (report.Verdict == FitVerdict.DoesNotFit && expertBytes > 0 && !settings.ExpertsOnDisk && ResolveDevice(settings.DeviceId)?.IsGpu == true)
        {
            var disk = settings with { CpuOffload = true, ExpertsOnDisk = true };
            var diskReport = Plan(report.Footprint, disk, memory);
            if (diskReport.ExpertsFromDisk)
            {
                suggestions.Add(new FitSuggestion($"Read experts from disk · about {ByteFormat.Format(DiskBytes(diskReport))} stay on disk, slow", diskReport.Verdict, disk));
            }
        }

        var shorter = new List<LoadSettings>();
        for (int context = settings.ContextLength / 2, i = 0; contexts is null && context >= MinSuggestedContext && i < ContextSteps; context /= 2, i++)
        {
            shorter.Add(settings with { ContextLength = context });
        }

        await AddBestAsync(suggestions, request, report, memory, shorter,
            (candidate, fp) => $"Context of {candidate.ContextLength.ToString("N0", CultureInfo.InvariantCulture)} tokens · needs about {ByteFormat.Format(fp.Total)}",
            cancellationToken).ConfigureAwait(false);

        string[] smallerKv = settings.KvCacheType switch
        {
            "f32" => ["f16", "q8_0", "q4_0"],
            "f16" => ["q8_0", "q4_0"],
            "q8_0" or "f8e4m3_b32" => ["q4_0"],
            _ => [],
        };
        await AddBestAsync(suggestions, request, report, memory, [.. smallerKv.Select(type => settings with { KvCacheType = type })],
            (candidate, fp) => $"{candidate.KvCacheType.ToUpperInvariant()} KV cache · needs about {ByteFormat.Format(fp.Total)}", cancellationToken)
            .ConfigureAwait(false);

        if (settings.Slots > 1)
        {
            await AddBestAsync(suggestions, request, report, memory, [settings with { Slots = 1 }],
                (_, fp) => $"One parallel slot · needs about {ByteFormat.Format(fp.Total)}", cancellationToken).ConfigureAwait(false);
        }

        var device = ResolveDevice(settings.DeviceId);
        if (device is { IsGpu: true, IsSplit: false } && _runtime.Devices.FirstOrDefault(option => option.IsSplit) is { } split)
        {
            var splitSettings = settings with { DeviceId = split.Id };
            var splitReport = Plan(report.Footprint, splitSettings, memory);
            if (splitReport.Verdict < report.Verdict)
            {
                suggestions.Add(new FitSuggestion($"Split over all {memory.Count(entry => entry.IsGpu)} GPUs", splitReport.Verdict, splitSettings));
            }
        }

        if (entry is not null)
        {
            AddSmallerVariant(suggestions, report, settings, memory, entry, library);
        }

        return suggestions;
    }

    /// <summary>
    /// The longest contexts that fit <paramref name="request"/>'s model with its other settings: on the chosen device, and (on a GPU with
    /// system memory to spare) with CPU offload. Every candidate is sized by the engine (<see cref="IModelRuntime.EstimateFootprintAsync"/>);
    /// a binary search over <see cref="ContextSizer.Candidates"/> needs a few estimates only.
    /// </summary>
    /// <param name="request">The load as configured.</param>
    /// <param name="memory">The devices' memory.</param>
    /// <param name="maxContext">The longest context to try (<see cref="ContextSizer.Limit"/>).</param>
    /// <param name="expertBytes">The routed experts' bytes (<see cref="IModelRuntime.EstimateExpertBytesAsync"/>), 0 for a dense model.</param>
    /// <param name="cancellationToken">Cancels the estimates.</param>
    public async Task<ContextFit> SizeContextAsync(ModelLoadRequest request, IReadOnlyList<DeviceMemory> memory, int maxContext, long expertBytes,
        CancellationToken cancellationToken)
    {
        var settings = request.Settings;
        var footprints = new Dictionary<int, ModelFootprint>();
        async Task<ModelFootprint> EstimateAsync(int context)
        {
            if (!footprints.TryGetValue(context, out var fp))
            {
                fp = await _runtime.EstimateFootprintAsync(request with { Settings = settings with { ContextLength = context } }, cancellationToken).ConfigureAwait(false);
                footprints[context] = fp;
            }

            return fp;
        }

        var candidates = ContextSizer.Candidates(maxContext, settings.ContextLength);
        var onDevice = settings with { CpuOffload = false };
        int? longest = await ContextSizer.LongestAsync(candidates,
            async context => Plan(await EstimateAsync(context).ConfigureAwait(false), onDevice with { ContextLength = context }, memory).FitsOnDevice).ConfigureAwait(false);
        ModelFootprint? longestFootprint = longest is { } l ? footprints[l] : null;

        var device = ResolveDevice(settings.DeviceId);
        bool canOffload = device?.IsGpu == true && memory.Any(entry => !entry.IsGpu);
        var longer = candidates.Where(context => context > (longest ?? 0)).ToList();
        if (!canOffload || longer.Count == 0)
        {
            return new ContextFit(maxContext, longest, longestFootprint);
        }

        // Contexts that fit memory: experts on disk would make every context "load".
        var offload = settings with { CpuOffload = true, ExpertsOnDisk = false };
        int? withOffload = await ContextSizer.LongestAsync(longer,
            async context => Plan(await EstimateAsync(context).ConfigureAwait(false), offload with { ContextLength = context }, memory).LoadsAsConfigured).ConfigureAwait(false);
        if (withOffload is not { } w)
        {
            return new ContextFit(maxContext, longest, longestFootprint);
        }

        var report = Plan(footprints[w], offload with { ContextLength = w }, memory);
        return new ContextFit(maxContext, longest, longestFootprint, w, footprints[w], IsExpertOffload(report, expertBytes));
    }

    /// <summary>
    /// Rows that show how far the configured load could go: the longest context on the device and with CPU offload (when they differ from
    /// the configured one), and the other KV cache type (F16 against Q8_0) when the model fits as configured and that type changes its size.
    /// </summary>
    /// <param name="request">The load as configured.</param>
    /// <param name="report">Its plan.</param>
    /// <param name="contexts">The longest contexts (<see cref="SizeContextAsync"/>).</param>
    /// <param name="memory">The devices' memory the plan used.</param>
    /// <param name="cancellationToken">Cancels the estimates.</param>
    public async Task<IReadOnlyList<FitSuggestion>> SizingAsync(ModelLoadRequest request, FitReport report, ContextFit contexts,
        IReadOnlyList<DeviceMemory> memory, CancellationToken cancellationToken)
    {
        var settings = request.Settings;
        var rows = new List<FitSuggestion>();
        string target = report.Devices.Count switch
        {
            1 when report.Devices[0].Device.IsGpu => $"on {report.Devices[0].Device.Name}",
            1 => "in system memory",
            var count and > 1 => $"on {count} GPUs",
            _ => "on the device",
        };
        if (contexts.OnDevice is { } longest && contexts.OnDeviceFootprint is { } longestFp && longest != settings.ContextLength)
        {
            rows.Add(new FitSuggestion($"Up to {Tokens(longest)} tokens {target} · needs about {ByteFormat.Format(longestFp.Total)}",
                Plan(longestFp, settings with { ContextLength = longest, CpuOffload = false }, memory).Verdict,
                settings with { ContextLength = longest, CpuOffload = false }));
        }

        if (contexts.WithOffload is { } offload && contexts.WithOffloadFootprint is { } offloadFp && offload != settings.ContextLength)
        {
            string kind = contexts.ExpertOffload ? "CPU expert offload" : "CPU offload";
            rows.Add(new FitSuggestion($"Up to {Tokens(offload)} tokens with {kind} · needs about {ByteFormat.Format(offloadFp.Total)}",
                FitVerdict.NeedsOffload, settings with { ContextLength = offload, CpuOffload = true }));
        }

        if (report.FitsOnDevice && KvAlternative(settings.KvCacheType) is { } other)
        {
            var candidate = settings with { KvCacheType = other };
            try
            {
                var fp = await _runtime.EstimateFootprintAsync(request with { Settings = candidate }, cancellationToken).ConfigureAwait(false);
                long delta = report.Footprint.Total - fp.Total;
                // Caches whose rows are not whole 32-value blocks stay F16: then the type changes nothing worth showing.
                if (Math.Abs(delta) >= 16L << 20)
                {
                    string change = delta > 0 ? $"{ByteFormat.Format(delta)} less" : $"{ByteFormat.Format(-delta)} more";
                    rows.Add(new FitSuggestion($"{KvName(other)} KV cache · needs about {ByteFormat.Format(fp.Total)} ({change})",
                        Plan(fp, candidate, memory).Verdict, candidate));
                }
            }
            catch (Exception error) when (error is not (OperationCanceledException or OutOfMemoryException))
            {
                // No row without an estimate.
            }
        }

        return rows;
    }

    /// <summary>
    /// The offload of <paramref name="report"/> moves only routed experts: what goes to the CPU is no more than the experts' weights (the engine
    /// offloads experts first and keeps attention and the shared layers on the GPU).
    /// </summary>
    /// <param name="report">A plan with <see cref="FitVerdict.NeedsOffload"/>.</param>
    /// <param name="expertBytes">The routed experts' bytes, 0 for a dense model.</param>
    public static bool IsExpertOffload(FitReport report, long expertBytes)
        => expertBytes > 0 && report.Verdict == FitVerdict.NeedsOffload && report.SystemMemory is { } system && system.NeededBytes <= expertBytes;

    /// <summary>
    /// The read speed assumed for experts on disk (a fast NVMe drive, about 3 GB/s): the engine measures the drive only once the model is
    /// loaded (<c>ExpertStore.MeasuredReadBytesPerSecond</c>).
    /// </summary>
    public const double AssumedDiskBytesPerSecond = 3e9;

    /// <summary>The bytes of routed experts a plan leaves on disk: what goes to the CPU beyond the system memory available.</summary>
    /// <param name="report">A plan with <see cref="FitReport.ExpertsFromDisk"/>.</param>
    public static long DiskBytes(FitReport report)
        => report.ExpertsFromDisk && report.SystemMemory is { } system ? Math.Max(0, system.NeededBytes - system.AvailableBytes) : 0;

    /// <summary>
    /// A rough decode speed with experts on disk, in tokens per second: the drive's read speed over the bytes a token reads from disk (the
    /// experts-per-token share of the experts left there, every read a cache miss), so a lower bound; null when nothing is on disk or the share is unknown.
    /// </summary>
    /// <param name="report">The plan.</param>
    /// <param name="activeShare">Experts per token / experts (<see cref="IModelRuntime.EstimateExpertActiveShareAsync"/>).</param>
    /// <param name="bytesPerSecond">The drive's read speed.</param>
    public static double? DiskTokensPerSecond(FitReport report, double activeShare, double bytesPerSecond = AssumedDiskBytesPerSecond)
    {
        long disk = DiskBytes(report);
        double perToken = disk * activeShare;
        return disk <= 0 || activeShare <= 0 || perToken <= 0 ? null : bytesPerSecond / perToken;
    }

    /// <summary>The KV cache type to compare with: Q8_0 for the default F16 cache, F16 for any other.</summary>
    private static string? KvAlternative(string type) => type switch
    {
        "f16" => "q8_0",
        "q8_0" or "q4_0" or "f8e4m3_b32" or "f32" => "f16",
        _ => null,
    };

    private static string KvName(string type) => type == "f16" ? "F16" : type.ToUpperInvariant();

    private static string Tokens(int context) => context.ToString("N0", CultureInfo.InvariantCulture);

    /// <summary>A smaller quantization of the same repository in the library: the largest that fits, else the smallest.</summary>
    private void AddSmallerVariant(List<FitSuggestion> suggestions, FitReport report, LoadSettings settings, IReadOnlyList<DeviceMemory> memory,
        LibraryEntry entry, IReadOnlyList<LibraryEntry> library)
    {
        if (entry.Manifest is not { } manifest || entry.SizeBytes <= 0)
        {
            return;
        }

        var smaller = library
            .Where(other => other.Manifest?.Repository == manifest.Repository && other.Path != entry.Path && other.SizeBytes > 0 && other.SizeBytes < entry.SizeBytes)
            .OrderByDescending(other => other.SizeBytes)
            .Select(other =>
            {
                // Weights scale with the file; the KV cache and activations stay.
                var fp = report.Footprint;
                var scaled = fp with { WeightBytes = (long)(fp.WeightBytes * ((double)other.SizeBytes / entry.SizeBytes)) };
                return (Entry: other, Footprint: scaled, Report: Plan(scaled, settings, memory));
            })
            .ToList();
        var best = smaller.FirstOrDefault(candidate => candidate.Report.FitsOnDevice);
        if (best.Entry is null)
        {
            best = smaller.Where(candidate => candidate.Report.Verdict < report.Verdict).OrderBy(candidate => candidate.Report.Verdict).FirstOrDefault();
        }

        if (best.Entry is not null)
        {
            var name = best.Entry.Quantization ?? best.Entry.Id;
            suggestions.Add(new FitSuggestion($"Load {name} instead ({ByteFormat.Format(best.Entry.SizeBytes)}) · needs about {ByteFormat.Format(best.Footprint.Total)}",
                best.Report.Verdict, Variant: best.Entry));
        }
    }

    /// <summary>
    /// Estimates the candidates in order and adds the first that fits on the device (counting automatic unloads), else the first that
    /// improves on the verdict.
    /// </summary>
    private async Task AddBestAsync(List<FitSuggestion> suggestions, ModelLoadRequest request, FitReport report, IReadOnlyList<DeviceMemory> memory,
        IReadOnlyList<LoadSettings> candidates, Func<LoadSettings, ModelFootprint, string> describe, CancellationToken cancellationToken)
    {
        FitSuggestion? improved = null;
        foreach (var candidate in candidates)
        {
            ModelFootprint fp;
            try
            {
                fp = await _runtime.EstimateFootprintAsync(request with { Settings = candidate }, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception error) when (error is not (OperationCanceledException or OutOfMemoryException))
            {
                continue;
            }

            var plan = Plan(fp, candidate, memory);
            if (plan.FitsOnDevice)
            {
                suggestions.Add(new FitSuggestion(describe(candidate, fp), plan.Verdict, candidate));
                return;
            }

            if (improved is null && plan.Verdict < report.Verdict)
            {
                improved = new FitSuggestion(describe(candidate, fp), plan.Verdict, candidate);
            }
        }

        if (improved is not null)
        {
            suggestions.Add(improved);
        }
    }

    private static ResidentUsage Usage(ILoadedModel model, IReadOnlyList<DeviceMemory> gpus)
    {
        IReadOnlyList<string> devices = model.Device.IsSplit ? [.. gpus.Select(gpu => gpu.Id)] : [model.Device.Id];
        bool resident = model.State is ResidentState.Ready or ResidentState.Busy;
        bool evictable = resident && !model.IsPinned && model.ActiveRequests == 0 && model.State == ResidentState.Ready;
        return new ResidentUsage(model.Id, devices, resident ? model.ResidentBytes : 0, evictable, model.LastUsed);
    }

    /// <summary><c>35 %</c> (with a non-breaking space), at least 1 %.</summary>
    public static string Percent(double share) => $"{Math.Max(1, Math.Round(share * 100)).ToString("F0", CultureInfo.InvariantCulture)}\u00A0%";
}
