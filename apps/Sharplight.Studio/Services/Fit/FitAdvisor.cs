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

    public FitAdvisor(IModelRuntime runtime, ISettingsService settings)
    {
        _runtime = runtime;
        _settings = settings;
    }

    /// <summary>Memory kept free on every device, in bytes.</summary>
    public long ReserveBytes => Math.Max(0, _settings.Current.Residency.ReserveMiB) * (1L << 20);

    /// <summary>The device a load with <paramref name="deviceId"/> goes to (the runtime's default when it names none it knows).</summary>
    /// <param name="deviceId">The device id of the load settings.</param>
    public ComputeDeviceOption? ResolveDevice(string? deviceId)
        => _runtime.Devices.FirstOrDefault(device => device.Id == deviceId) ?? _runtime.Devices.FirstOrDefault();

    /// <summary>The memory of every physical device now.</summary>
    /// <param name="cancellationToken">Cancels the query.</param>
    public Task<IReadOnlyList<DeviceMemory>> GetMemoryAsync(CancellationToken cancellationToken)
        => _runtime.GetDeviceMemoryAsync(_settings.Current.DisabledFeatures, cancellationToken);

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
    public async Task<IReadOnlyList<FitSuggestion>> SuggestAsync(ModelLoadRequest request, FitReport report, IReadOnlyList<DeviceMemory> memory,
        LibraryEntry? entry, IReadOnlyList<LibraryEntry> library, CancellationToken cancellationToken)
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

        var contexts = new List<LoadSettings>();
        for (int context = settings.ContextLength / 2, i = 0; context >= MinSuggestedContext && i < ContextSteps; context /= 2, i++)
        {
            contexts.Add(settings with { ContextLength = context });
        }

        await AddBestAsync(suggestions, request, report, memory, contexts,
            (candidate, fp) => $"Context of {candidate.ContextLength.ToString("N0", CultureInfo.InvariantCulture)} tokens · needs about {ByteFormat.Format(fp.Total)}",
            cancellationToken).ConfigureAwait(false);

        string[] smallerKv = settings.KvCacheType switch
        {
            "f32" => ["f16", "q8_0", "q4_0"],
            "f16" => ["q8_0", "q4_0"],
            "q8_0" => ["q4_0"],
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
