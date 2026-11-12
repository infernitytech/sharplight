using System.Collections;
using System.Collections.Immutable;
using System.Globalization;

namespace Sharplight.Graph;

/// <summary>Immutable bag of op attributes (ints, floats, strings, arrays, enums).</summary>
public sealed class Attributes : IReadOnlyDictionary<string, object>
{
    readonly ImmutableSortedDictionary<string, object> _values;

    public static Attributes Empty { get; } = new(ImmutableSortedDictionary<string, object>.Empty);

    Attributes(ImmutableSortedDictionary<string, object> values) => _values = values;

    public Attributes With(string key, object value) => new(_values.SetItem(key, value));

    public static Attributes From(params ReadOnlySpan<(string Key, object Value)> items)
    {
        var b = ImmutableSortedDictionary.CreateBuilder<string, object>(StringComparer.Ordinal);
        foreach (var (k, v) in items) b[k] = v;
        return new Attributes(b.ToImmutable());
    }

    public T Get<T>(string key) => _values.TryGetValue(key, out var v)
        ? Convert<T>(v, key)
        : throw new KeyNotFoundException($"Missing attribute '{key}'.");

    public T Get<T>(string key, T defaultValue) => _values.TryGetValue(key, out var v) ? Convert<T>(v, key) : defaultValue;

    public bool Has(string key) => _values.ContainsKey(key);

    static T Convert<T>(object v, string key)
    {
        if (v is T t) return t;
        try
        {
            return (T)System.Convert.ChangeType(v, typeof(T), CultureInfo.InvariantCulture);
        }
        catch (Exception e)
        {
            throw new InvalidCastException($"Attribute '{key}' ({v.GetType().Name}) is not convertible to {typeof(T).Name}.", e);
        }
    }

    public object this[string key] => _values[key];
    public IEnumerable<string> Keys => _values.Keys;
    public IEnumerable<object> Values => _values.Values;
    public int Count => _values.Count;
    public bool ContainsKey(string key) => _values.ContainsKey(key);
    public bool TryGetValue(string key, [System.Diagnostics.CodeAnalysis.MaybeNullWhen(false)] out object value) => _values.TryGetValue(key, out value);
    public IEnumerator<KeyValuePair<string, object>> GetEnumerator() => _values.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>Stable textual form, usable as part of a kernel cache key.</summary>
    public override string ToString() =>
        "{" + string.Join(", ", _values.Select(kv => $"{kv.Key}={Format(kv.Value)}")) + "}";

    static string Format(object v) => v switch
    {
        float f => f.ToString("R", CultureInfo.InvariantCulture),
        double d => d.ToString("R", CultureInfo.InvariantCulture),
        Array a => "[" + string.Join(",", a.Cast<object>().Select(Format)) + "]",
        _ => System.Convert.ToString(v, CultureInfo.InvariantCulture) ?? "",
    };
}
