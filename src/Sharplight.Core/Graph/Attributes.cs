using System.Collections;
using System.Collections.Immutable;
using System.Globalization;

namespace Sharplight.Graph;

/// <summary>Immutable bag of op attributes (ints, floats, strings, arrays, enums).</summary>
public sealed class Attributes : IReadOnlyDictionary<string, object>
{
    private readonly ImmutableSortedDictionary<string, object> _values;

    /// <summary>The attribute set without any entries.</summary>
    public static Attributes Empty { get; } = new(ImmutableSortedDictionary<string, object>.Empty);

    private Attributes(ImmutableSortedDictionary<string, object> values) => _values = values;

    /// <summary>
    /// Returns a copy of this set in which <paramref name="key"/> is set to <paramref name="value"/>; this instance is unchanged.
    /// </summary>
    /// <param name="key">Attribute name (case-sensitive).</param>
    /// <param name="value">Attribute value: an integer, float, bool, string or an array of those. Used as part of kernel cache keys, so it must have a stable textual form.</param>
    public Attributes With(string key, object value) => new(_values.SetItem(key, value));

    /// <summary>
    /// Creates a set from key/value pairs; a later pair overrides an earlier pair with the same key.
    /// </summary>
    /// <param name="items">The entries to store.</param>
    public static Attributes From(params ReadOnlySpan<(string Key, object Value)> items)
    {
        var b = ImmutableSortedDictionary.CreateBuilder<string, object>(StringComparer.Ordinal);
        foreach (var (k, v) in items)
        {
            b[k] = v;
        }

        return new Attributes(b.ToImmutable());
    }

    /// <summary>
    /// Reads an attribute, converting numeric values and arrays to <typeparamref name="T"/> (for example a stored <c>int[]</c> read as <c>long[]</c>).
    /// </summary>
    /// <param name="key">Attribute name.</param>
    /// <returns>The converted value.</returns>
    /// <exception cref="KeyNotFoundException">The attribute is not present.</exception>
    /// <exception cref="InvalidCastException">The stored value cannot be converted to <typeparamref name="T"/>.</exception>
    public T Get<T>(string key) => _values.TryGetValue(key, out var v)
        ? ValueConverter.Convert<T>(v, "Attribute", key)
        : throw new KeyNotFoundException($"Missing attribute '{key}'.");

    /// <summary>
    /// Reads an optional attribute, converting like <see cref="Get{T}(string)"/>.
    /// </summary>
    /// <param name="key">Attribute name.</param>
    /// <param name="defaultValue">Returned when the attribute is not present.</param>
    /// <returns>The converted value, or <paramref name="defaultValue"/>.</returns>
    /// <exception cref="InvalidCastException">The stored value cannot be converted to <typeparamref name="T"/>.</exception>
    public T Get<T>(string key, T defaultValue) => _values.TryGetValue(key, out var v) ? ValueConverter.Convert<T>(v, "Attribute", key) : defaultValue;

    /// <summary>Whether the attribute is present.</summary>
    public bool Has(string key) => _values.ContainsKey(key);

    /// <summary>Gets the raw stored value of an attribute.</summary>
    public object this[string key] => _values[key];
    /// <summary>The attribute names in ordinal sort order.</summary>
    public IEnumerable<string> Keys => _values.Keys;
    /// <summary>The raw attribute values, in the order of <see cref="Keys"/>.</summary>
    public IEnumerable<object> Values => _values.Values;
    /// <summary>Number of attributes.</summary>
    public int Count => _values.Count;
    /// <summary>Whether the attribute is present (same as <see cref="Has(string)"/>).</summary>
    public bool ContainsKey(string key) => _values.ContainsKey(key);
    /// <summary>Gets the raw stored value of an attribute if it is present.</summary>
    public bool TryGetValue(string key, [System.Diagnostics.CodeAnalysis.MaybeNullWhen(false)] out object value) => _values.TryGetValue(key, out value);
    /// <summary>Enumerates the entries in ordinal key order.</summary>
    public IEnumerator<KeyValuePair<string, object>> GetEnumerator() => _values.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>Stable textual form, usable as part of a kernel cache key.</summary>
    public override string ToString() =>
        "{" + string.Join(", ", _values.Select(kv => $"{kv.Key}={Format(kv.Value)}")) + "}";

    private static string Format(object v) => v switch
    {
        float f => f.ToString("R", CultureInfo.InvariantCulture),
        double d => d.ToString("R", CultureInfo.InvariantCulture),
        Array a => "[" + string.Join(",", a.Cast<object>().Select(Format)) + "]",
        _ => System.Convert.ToString(v, CultureInfo.InvariantCulture) ?? "",
    };
}
