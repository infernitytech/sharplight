using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Package;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>The metadata tab: every model property and source metadata entry, filterable, with the selected value in full.</summary>
internal sealed partial class MetadataViewModel : ObservableObject
{
    /// <summary>Source label of model properties.</summary>
    public const string PropertySource = "property";

    /// <summary>Source label of source metadata.</summary>
    public const string MetadataSource = "metadata";

    private readonly IReadOnlyList<KeyValueRow> _all;

    /// <param name="definition">The decoded definition, or null when it does not decode.</param>
    public MetadataViewModel(DefinitionInfo? definition)
    {
        _all = definition is null ? [] : [.. Rows(PropertySource, definition.Properties), .. Rows(MetadataSource, definition.Metadata)];
        Values = _all;
        Summary = $"{definition?.Properties.Count ?? 0} properties, {definition?.Metadata.Count ?? 0} metadata entries";
    }

    /// <summary>Number of entries before filtering.</summary>
    public int Count => _all.Count;

    /// <summary>Counts per source.</summary>
    public string Summary { get; }

    /// <summary>Filter (whitespace-separated terms, all must match key, value or source).</summary>
    [ObservableProperty]
    public partial string? Filter { get; set; }

    /// <summary>The entries matching <see cref="Filter"/>.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<KeyValueRow> Values { get; private set; }

    /// <summary>The entry whose full value is shown.</summary>
    [ObservableProperty]
    public partial KeyValueRow? SelectedValue { get; set; }

    partial void OnFilterChanged(string? value) => Values = TextFilter.Apply(_all, value);

    private static IEnumerable<KeyValueRow> Rows(string source, IReadOnlyDictionary<string, object> values) =>
        values.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => new KeyValueRow(source, kv.Key, kv.Value));
}
