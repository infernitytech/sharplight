using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Package;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>The artifacts tab: every stored artifact set (kernels, pipeline caches) ordered by provider and tier, and the selected one.</summary>
internal sealed partial class ArtifactsViewModel : ObservableObject
{
    /// <param name="sets">The stored sets in section-table order.</param>
    public ArtifactsViewModel(IReadOnlyList<ArtifactSetInfo> sets)
    {
        Sets = [.. sets.Select(s => new ArtifactSetViewModel(s))
            .OrderBy(s => s.Provider, StringComparer.Ordinal).ThenBy(s => s.Tier, StringComparer.Ordinal).ThenBy(s => s.Id, StringComparer.Ordinal)];
        Summary = Sets.Count == 0
            ? "No precompiled artifacts: kernels are compiled when the model is loaded."
            : $"{ByteFormat.Count(Sets.Count, "artifact set", "artifact sets")} for {ByteFormat.Count(Sets.Select(s => s.Provider).Distinct().Count(), "provider", "providers")}, "
                + $"{ByteFormat.Count(Sets.Sum(s => s.EntryCount), "entry", "entries")}, {ByteFormat.Format(Sets.Sum(s => s.TotalBytes))}";
        SelectedSet = Sets.FirstOrDefault();
    }

    /// <summary>Every set, by provider, tier and id.</summary>
    public IReadOnlyList<ArtifactSetViewModel> Sets { get; }

    /// <summary>True when the package holds no artifacts.</summary>
    public bool IsEmpty => Sets.Count == 0;

    /// <summary>Totals.</summary>
    public string Summary { get; }

    /// <summary>The set shown on the right.</summary>
    [ObservableProperty]
    public partial ArtifactSetViewModel? SelectedSet { get; set; }

    /// <summary>Selects the set stored in the section at <paramref name="offset"/>.</summary>
    public void Select(long offset)
    {
        if (Sets.FirstOrDefault(s => s.SectionOffset == offset) is { } set)
        {
            SelectedSet = set;
        }
    }
}
