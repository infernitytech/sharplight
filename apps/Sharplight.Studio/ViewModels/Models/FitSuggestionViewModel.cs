using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services.Fit;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A suggested change in a fit estimate, applied with a click.</summary>
internal sealed partial class FitSuggestionViewModel
{
    private readonly Action<FitSuggestion> _apply;

    public FitSuggestionViewModel(FitSuggestion suggestion, Action<FitSuggestion> apply)
    {
        Suggestion = suggestion;
        _apply = apply;
    }

    /// <summary>The change.</summary>
    public FitSuggestion Suggestion { get; }

    /// <summary>What to change and what it needs then.</summary>
    public string Text => Suggestion.Text;

    /// <summary>The verdict with the change, e.g. <c>fits</c>.</summary>
    public string VerdictText => Suggestion.Verdict switch
    {
        FitVerdict.Fits => "fits",
        FitVerdict.FitsAfterUnloading => "after unloading",
        FitVerdict.NeedsOffload => "with offload",
        _ => "still too large",
    };

    /// <summary>With the change, it fits without offloading or unloading by hand.</summary>
    public bool Fits => Suggestion.Verdict is FitVerdict.Fits or FitVerdict.FitsAfterUnloading;

    /// <summary>Applies the change to the load panel.</summary>
    [RelayCommand]
    private void Apply() => _apply(Suggestion);
}
