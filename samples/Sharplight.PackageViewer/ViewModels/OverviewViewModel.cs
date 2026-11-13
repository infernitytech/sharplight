using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Package;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>The overview tab: file facts, section table, model properties and metadata.</summary>
public sealed partial class OverviewViewModel : ObservableObject
{
    private readonly IReadOnlyList<KeyValueRow> _allValues;

    /// <param name="info">The package.</param>
    /// <param name="definition">Its decoded definition, or null when it does not decode.</param>
    /// <param name="definitionError">Why the definition did not decode.</param>
    public OverviewViewModel(PackageInfo info, DefinitionInfo? definition, string? definitionError)
    {
        Path = info.Path;
        FormatVersion = info.Version;
        FileSize = ByteSizeFormat.FormatExact(info.FileSize);
        Reclaimable = info.ReclaimableBytes > 0 ? $"{ByteSizeFormat.FormatExact(info.ReclaimableBytes)} (sharplight-convert compact)" : "none";
        Weights = $"{info.Weights.Count} ({ByteSizeFormat.Format(info.WeightBytes)})";
        ArtifactSets = info.ArtifactSets.Count == 0 ? "none" : $"{info.ArtifactSets.Count} ({ByteSizeFormat.Format(info.ArtifactSets.Sum(s => s.TotalBytes))})";
        Sections = [.. info.Sections.Select(SectionRow.From)];
        DefinitionError = definitionError;
        if (definition is null)
        {
            Model = "(definition not readable)";
            Graphs = "";
            _allValues = [];
        }
        else
        {
            Model = $"{definition.Name} ({definition.Architecture}), {definition.States.Count} state(s)";
            Graphs = string.Join(", ", definition.Graphs.Select(g => $"{g.Name} ({g.NodeCount} nodes)"));
            _allValues = [.. Rows("property", definition.Properties), .. Rows("metadata", definition.Metadata)];
        }

        Values = _allValues;
    }

    public string Path { get; }

    public int FormatVersion { get; }

    public string FileSize { get; }

    public string Reclaimable { get; }

    public string Weights { get; }

    public string ArtifactSets { get; }

    public string Model { get; }

    public string Graphs { get; }

    public string? DefinitionError { get; }

    public IReadOnlyList<SectionRow> Sections { get; }

    /// <summary>Number of properties and metadata entries before filtering.</summary>
    public int ValueCount => _allValues.Count;

    /// <summary>Filter of the property/metadata table (whitespace-separated terms, all must match key, value or source).</summary>
    [ObservableProperty]
    public partial string? Filter { get; set; }

    /// <summary>The properties and metadata matching <see cref="Filter"/>.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<KeyValueRow> Values { get; private set; }

    /// <summary>The row whose full value the detail box shows.</summary>
    [ObservableProperty]
    public partial KeyValueRow? SelectedValue { get; set; }

    partial void OnFilterChanged(string? value) => Values = TextFilter.Apply(_allValues, value);

    private static IEnumerable<KeyValueRow> Rows(string source, IReadOnlyDictionary<string, object> values) =>
        values.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => new KeyValueRow(source, kv.Key, kv.Value));
}
