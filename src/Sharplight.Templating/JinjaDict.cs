using System.Collections;

namespace Sharplight.Templating;

/// <summary>Insertion-ordered dictionary (Python dict semantics). Keys are strings, longs or bools.</summary>
public sealed class JinjaDict : IEnumerable<KeyValuePair<object, object?>>
{
    private readonly Dictionary<object, object?> _map = new();
    private readonly List<object> _keys = new();

    /// <summary>Number of entries.</summary>
    public int Count => _keys.Count;
    /// <summary>Keys in insertion order.</summary>
    public IReadOnlyList<object> Keys => _keys;

    /// <summary>Gets or sets the value for a key; setting an existing key keeps its position. Getting a missing key throws <see cref="KeyNotFoundException"/>.</summary>
    public object? this[object key]
    {
        get => _map[key];
        set
        {
            key = NormalizeKey(key);
            if (!_map.ContainsKey(key))
            {
                _keys.Add(key);
            }

            _map[key] = value;
        }
    }

    internal static object NormalizeKey(object key) => key switch
    {
        int i => (long)i,
        short s => (long)s,
        byte b => (long)b,
        _ => key,
    };

    /// <summary>Looks up a key; integral keys are normalized to <see cref="long"/>.</summary>
    public bool TryGetValue(object key, out object? value) => _map.TryGetValue(NormalizeKey(key), out value);
    /// <summary>True if the key is present.</summary>
    public bool ContainsKey(object key) => _map.ContainsKey(NormalizeKey(key));

    /// <summary>Removes a key; returns false if it was not present.</summary>
    public bool Remove(object key)
    {
        key = NormalizeKey(key);
        if (!_map.Remove(key))
        {
            return false;
        }

        _keys.Remove(key);
        return true;
    }

    /// <summary>Removes all entries.</summary>
    public void Clear()
    {
        _map.Clear();
        _keys.Clear();
    }

    /// <summary>Shallow copy preserving insertion order.</summary>
    public JinjaDict Clone()
    {
        var d = new JinjaDict();
        foreach (var k in _keys)
        {
            d[k] = _map[k];
        }

        return d;
    }

    /// <summary>Enumerates entries in insertion order.</summary>
    public IEnumerator<KeyValuePair<object, object?>> GetEnumerator()
    {
        foreach (var k in _keys.ToArray())
        {
            yield return new KeyValuePair<object, object?>(k, _map[k]);
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
