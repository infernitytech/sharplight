using Sharplight.Package;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>A model property, metadata entry or descriptor pair: the value shortened for the list, in full on demand.</summary>
internal sealed class KeyValueRow : IFilterableRow
{
    private const int TableItems = 8;
    private const int TableLength = 160;
    private const int DetailItems = 4096;
    private const int DetailLength = 1 << 20;

    private readonly object _raw;

    /// <param name="source">Where the pair comes from (<c>property</c>, <c>metadata</c>, or empty).</param>
    /// <param name="key">The key.</param>
    /// <param name="value">The raw value (scalar, string or array).</param>
    public KeyValueRow(string source, string key, object value)
    {
        Source = source;
        Key = key;
        _raw = value;
        Value = DefinitionInfo.FormatValue(value, TableItems, TableLength);
    }

    /// <summary>Where the pair comes from (<c>property</c>, <c>metadata</c>, or empty for descriptors).</summary>
    public string Source { get; }

    /// <summary>The key.</summary>
    public string Key { get; }

    /// <summary>The value shortened to one line.</summary>
    public string Value { get; }

    /// <summary>The value with (almost) all items and characters, for the detail box; formatted on access.</summary>
    public string Detail => _raw is string text ? text : DefinitionInfo.FormatValue(_raw, DetailItems, DetailLength);

    /// <inheritdoc/>
    public bool Matches(string[] terms) => TextFilter.Matches(terms, Key, Value, Source);
}
