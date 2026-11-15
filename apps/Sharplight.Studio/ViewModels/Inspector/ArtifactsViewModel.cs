using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Package;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>
/// The artifacts tab: the precompilation stages stored per provider (portable code per device profile, device code per device and
/// driver), every stored artifact set ordered by provider and stage, and the selected one.
/// </summary>
internal sealed partial class ArtifactsViewModel : ObservableObject
{
    /// <param name="sets">The stored sets in section-table order.</param>
    public ArtifactsViewModel(IReadOnlyList<ArtifactSetInfo> sets)
    {
        Sets = [.. sets.Select(s => new ArtifactSetViewModel(s))
            .OrderBy(s => s.Provider, StringComparer.Ordinal).ThenBy(s => s.StageOrder).ThenBy(s => s.Id, StringComparer.Ordinal)];
        Stages = [.. Sets.Select(s => new ArtifactStageRow(s.Provider, s.Stage, s.BuiltFor, s.Label))];
        Summary = Sets.Count == 0
            ? "No precompiled artifacts: kernels are compiled when the model is loaded."
            : $"{ByteFormat.Count(Sets.Count, "artifact set", "artifact sets")} for {ByteFormat.Count(Sets.Select(s => s.Provider).Distinct().Count(), "provider", "providers")}, "
                + $"{ByteFormat.Count(Sets.Sum(s => s.EntryCount), "entry", "entries")}, {ByteFormat.Format(Sets.Sum(s => s.TotalBytes))}";
        SelectedSet = Sets.FirstOrDefault();
    }

    /// <summary>Every set, by provider, tier and id.</summary>
    public IReadOnlyList<ArtifactSetViewModel> Sets { get; }

    /// <summary>The stages per provider and target, in the order of <see cref="Sets"/>.</summary>
    public IReadOnlyList<ArtifactStageRow> Stages { get; }

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
