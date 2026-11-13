using Sharplight.Package;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>A model property, metadata entry or descriptor pair: the value shortened for the table, in full on demand.</summary>
public sealed class KeyValueRow : IFilterableRow
{
    private const int TableItems = 8;
    private const int TableLength = 160;
    private const int DetailItems = 4096;
    private const int DetailLength = 1 << 20;

    private readonly object _raw;

    public KeyValueRow(string source, string key, object value)
    {
        Source = source;
        Key = key;
        _raw = value;
        Value = DefinitionInfo.FormatValue(value, TableItems, TableLength);
    }

    /// <summary>Where the pair comes from (<c>property</c>, <c>metadata</c>, or empty for descriptors).</summary>
    public string Source { get; }

    public string Key { get; }

    /// <summary>The value shortened to one table line.</summary>
    public string Value { get; }

    /// <summary>The value with (almost) all items and characters, for the detail box; formatted on access.</summary>
    public string Detail => _raw is string text ? text : DefinitionInfo.FormatValue(_raw, DetailItems, DetailLength);

    public bool Matches(string[] terms) => TextFilter.Matches(terms, Key, Value, Source);
}
