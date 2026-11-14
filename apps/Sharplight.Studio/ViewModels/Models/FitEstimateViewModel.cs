using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Fit;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>The memory estimate in the load panel: verdict, footprint, each device's part and suggested changes.</summary>
internal sealed class FitEstimateViewModel
{
    public FitEstimateViewModel(FitReport report, IReadOnlyList<FitSuggestionViewModel> suggestions)
    {
        Report = report;
        Suggestions = suggestions;
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

    /// <summary>The verdict as a heading.</summary>
    public string Title => FitText.Title(Report);

    /// <summary>What it means.</summary>
    public string Message => FitText.Message(Report, FitText.Target(Report));

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
