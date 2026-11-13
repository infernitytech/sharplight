using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Package;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>The artifacts tab: stored sets grouped by provider, then tier; the selected set's descriptor and entries.</summary>
public sealed partial class ArtifactsViewModel : ObservableObject
{
    public ArtifactsViewModel(IReadOnlyList<ArtifactSetInfo> sets)
    {
        var all = sets.Select(s => new ArtifactSetViewModel(s)).ToList();
        Sets = all;
        Providers = [.. all.GroupBy(s => s.Provider).OrderBy(g => g.Key, StringComparer.Ordinal).Select(Provider)];
        Summary = all.Count == 0
            ? "No artifact sets: kernels are compiled when the model is loaded (add some with sharplight-convert precompile)."
            : $"{all.Count} artifact set(s), {ByteSizeFormat.Format(all.Sum(s => s.TotalBytes))}";
    }

    /// <summary>All sets in section-table order.</summary>
    public IReadOnlyList<ArtifactSetViewModel> Sets { get; }

    /// <summary>Tree roots: one node per provider, with one child per tier, with one leaf per set.</summary>
    public IReadOnlyList<ArtifactTreeNode> Providers { get; }

    public string Summary { get; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SelectedSet))]
    public partial ArtifactTreeNode? SelectedNode { get; set; }

    /// <summary>The set of the selected leaf, or null when a provider or tier is selected.</summary>
    public ArtifactSetViewModel? SelectedSet => SelectedNode?.Set;

    private static ArtifactTreeNode Provider(IGrouping<string, ArtifactSetViewModel> provider)
    {
        var tiers = provider.GroupBy(s => s.Tier).OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(tier => new ArtifactTreeNode(tier.Key, Totals(tier), [.. tier.Select(Leaf)], null));
        return new ArtifactTreeNode(provider.Key, Totals(provider), [.. tiers], null);
    }

    private static ArtifactTreeNode Leaf(ArtifactSetViewModel set) =>
        new(set.IsLegacy ? $"{set.Label} (version 1)" : set.Label, set.DistinguishingDescriptor, [], set);

    private static string Totals(IEnumerable<ArtifactSetViewModel> sets)
    {
        var list = sets.ToList();
        return $"{list.Count} set(s), {ByteSizeFormat.Format(list.Sum(s => s.TotalBytes))}";
    }
}
