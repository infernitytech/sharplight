using Sharplight.Hosting.Fit;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Fit;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>
/// The memory estimate in the load panel: verdict, footprint, each device's part, how far the context could go (and what the other KV cache
/// type changes), and suggested changes.
/// </summary>
internal sealed class FitEstimateViewModel
{
    /// <param name="report">The plan of the load as configured.</param>
    /// <param name="suggestions">Changes that make it fit.</param>
    /// <param name="sizing">The longest contexts and the other KV cache type (<see cref="Services.Fit.FitAdvisor.SizingAsync"/>).</param>
    /// <param name="contexts">The longest contexts, once sized.</param>
    /// <param name="expertOffload">The offload moves routed experts only.</param>
    /// <param name="diskTokensPerSecond">The decode speed estimate with experts on disk, or null.</param>
    public FitEstimateViewModel(FitReport report, IReadOnlyList<FitSuggestionViewModel> suggestions, IReadOnlyList<FitSuggestionViewModel>? sizing = null,
        ContextFit? contexts = null, bool expertOffload = false, double? diskTokensPerSecond = null)
    {
        DiskTokensPerSecond = diskTokensPerSecond;
        Report = report;
        Suggestions = suggestions;
        Sizing = sizing ?? [];
        Contexts = contexts;
        ExpertOffload = expertOffload;
        var rows = report.Devices.Select(device => new DeviceFitViewModel(device)).ToList();
        if (report.SystemMemory is { } system)
        {
            rows.Add(new DeviceFitViewModel(system, offloaded: true));
        }

        Devices = rows;
    }

    /// <summary>The plan.</summary>
    public FitReport Report { get; }

    /// <summary>The verdict.</summary>
    public FitVerdict Verdict => Report.Verdict;

    /// <summary>The offload moves routed experts only (a mixture-of-experts model).</summary>
    public bool ExpertOffload { get; }

    /// <summary>The verdict as a heading.</summary>
    public string Title => FitText.Title(Report, ExpertOffload);

    /// <summary>What it means.</summary>
    public string Message => FitText.Message(Report, FitText.Target(Report), ExpertOffload, DiskTokensPerSecond);

    /// <summary>The decode speed estimate with experts on disk, or null.</summary>
    public double? DiskTokensPerSecond { get; }

    /// <summary>Routed experts stay on disk: it runs, slowly.</summary>
    public bool FromDisk => Report.ExpertsFromDisk;

    /// <summary>The longest contexts that fit, once sized.</summary>
    public ContextFit? Contexts { get; }

    /// <summary>
    /// One line on the context: <c>32,768 tokens is the longest context that fits on the GPU.</c>, <c>No context fits on the GPU.</c>, or empty
    /// when not sized yet or when a row of <see cref="Sizing"/> says it.
    /// </summary>
    public string ContextSummary
    {
        get
        {
            if (Contexts is not { } contexts)
            {
                return "";
            }

            string where = Report.Devices.Count == 1 && !Report.Devices[0].Device.IsGpu ? "in system memory" : "on the GPU";
            if (contexts.OnDevice is not { } longest)
            {
                return $"Not even {Tokens(Math.Min(contexts.MaxContext, ContextSizer.MinContext))} tokens of context fit {where}.";
            }

            if (longest != ConfiguredContext)
            {
                return "";
            }

            return longest == contexts.MaxContext
                ? $"The model's full context ({Tokens(longest)} tokens) fits {where}."
                : $"{Tokens(longest)} tokens is the longest context that fits {where}.";
        }
    }

    /// <summary>The configured context (set by the panel), for <see cref="ContextSummary"/>.</summary>
    public int ConfiguredContext { get; init; }

    /// <summary>There is a context line.</summary>
    public bool HasContextSummary => ContextSummary.Length > 0;

    /// <summary>The longest contexts on the device and with offload, and the other KV cache type, each applied with a click.</summary>
    public IReadOnlyList<FitSuggestionViewModel> Sizing { get; }

    private static string Tokens(int context) => context.ToString("N0", System.Globalization.CultureInfo.InvariantCulture);

    /// <summary>There are sizing rows.</summary>
    public bool HasSizing => Sizing.Count > 0;

    /// <summary>It loads as configured.</summary>
    public bool IsGood => FitText.IsGood(Report);

    /// <summary>It loads only after a change (unloading pinned models, turning on offload) or runs partly on the CPU.</summary>
    public bool IsWarning => !IsGood && !IsBad;

    /// <summary>It does not fit.</summary>
    public bool IsBad => FitText.IsBad(Report);

    /// <summary><c>≈ 9.1 GB</c>.</summary>
    public string TotalText => "≈ " + ByteFormat.Format(Report.Footprint.Total);

    /// <summary>Weights, KV cache and activations.</summary>
    public string Breakdown => FitText.Breakdown(Report.Footprint);

    /// <summary>Each device's part (and system memory when offloading).</summary>
    public IReadOnlyList<DeviceFitViewModel> Devices { get; }

    /// <summary>Changes that make it fit.</summary>
    public IReadOnlyList<FitSuggestionViewModel> Suggestions { get; }

    /// <summary>There are suggestions.</summary>
    public bool HasSuggestions => Suggestions.Count > 0;
}
