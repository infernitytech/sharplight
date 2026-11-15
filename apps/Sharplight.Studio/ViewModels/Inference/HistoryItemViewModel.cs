using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Studio.Services.Pages;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>A run in the history list.</summary>
internal sealed partial class HistoryItemViewModel : ObservableObject
{
    internal HistoryItemViewModel(RunRecord record)
    {
        Record = record;
    }

    /// <summary>The run.</summary>
    public RunRecord Record { get; }

    /// <summary>What the run was for.</summary>
    public string Title => Record.Title;

    /// <summary>When it ran, and on which model.</summary>
    public string Detail => $"{Record.Created.LocalDateTime.ToString("g", System.Globalization.CultureInfo.CurrentCulture)} · {Record.Model} · {Record.Seconds:0.0} s";

    /// <summary>How it ended, in a word.</summary>
    public string OutcomeText => Record.Outcome switch
    {
        RunOutcome.Completed => "Done",
        RunOutcome.Failed => "Failed",
        _ => "Cancelled",
    };

    /// <summary>The run failed.</summary>
    public bool IsFailed => Record.Outcome == RunOutcome.Failed;

    /// <summary>The failure message, or an excerpt of the output.</summary>
    public string Summary => Record.Error ?? Record.Preview ?? "";

    /// <summary>The run's output can be shown again (it completed).</summary>
    public bool CanRestore => Record.Outcome == RunOutcome.Completed;

    /// <summary>Delete was clicked once; the next click deletes.</summary>
    [ObservableProperty]
    public partial bool ConfirmDelete { get; set; }
}
