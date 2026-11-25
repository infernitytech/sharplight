using System.Globalization;
using Sharplight.Hosting.Fit;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Fit;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>The approximate fit of a model that is not downloaded yet, as a badge and a sentence for the download confirmation.</summary>
internal sealed class FitBadgeViewModel
{
    /// <param name="report">The plan for the default load settings.</param>
    /// <param name="split">The plan splits the model over every GPU.</param>
    /// <param name="shapeKnown">The KV cache was sized from the model's dimensions (else from a share of the weights).</param>
    /// <param name="context">The context the estimate assumes.</param>
    public FitBadgeViewModel(FitReport report, bool split, bool shapeKnown, int context)
    {
        Report = report;
        IsSplit = split;
        ShapeKnown = shapeKnown;
        Context = context;
    }

    /// <summary>The plan.</summary>
    public FitReport Report { get; }

    /// <summary>The plan splits the model over every GPU.</summary>
    public bool IsSplit { get; }

    /// <summary>The KV cache was sized from the model's dimensions.</summary>
    public bool ShapeKnown { get; }

    /// <summary>The context the estimate assumes.</summary>
    public int Context { get; }

    /// <summary>Short verdict: <c>Fits</c>, <c>Fits on 2 GPUs</c>, <c>CPU offload ~40 %</c>, <c>Too large</c>.</summary>
    public string Text => Report.Verdict switch
    {
        FitVerdict.Fits when IsSplit => $"Fits on {Report.Devices.Count} GPUs",
        FitVerdict.Fits => "Fits",
        FitVerdict.FitsAfterUnloading => "Fits after unloading",
        FitVerdict.NeedsOffload when Report.ExpertsFromDisk => "From disk (slow)",
        FitVerdict.NeedsOffload => $"CPU offload ~{FitAdvisor.Percent(Report.CpuShare)}",
        _ => "Too large",
    };

    /// <summary>Loads with the default settings (counting automatic unloads).</summary>
    public bool IsGood => FitText.IsGood(Report);

    /// <summary>Loads only partly on the CPU or after unloading pinned models.</summary>
    public bool IsWarning => !IsGood && !IsBad;

    /// <summary>Does not fit, not even with CPU offload.</summary>
    public bool IsBad => FitText.IsBad(Report);

    /// <summary>The verdict as a heading, marked approximate.</summary>
    public string Title => FitText.Title(Report) + " (approximate)";

    /// <summary>What the estimate assumes and what the verdict means.</summary>
    public string Detail
    {
        get
        {
            string context = Context.ToString("N0", CultureInfo.InvariantCulture);
            string basis = ShapeKnown ? "the model's dimensions" : "the download size only";
            return $"{FitText.Message(Report, FitText.Target(Report))} Estimated from {basis} for the default load settings ({context} tokens of context): "
                + $"{FitText.Breakdown(Report.Footprint)}.";
        }
    }

    /// <summary>Tooltip of the badge.</summary>
    public string ToolTip => $"About {ByteFormat.Format(Report.Footprint.Total)} needed (approximate). {Detail}";
}
