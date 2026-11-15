namespace Sharplight.Import.PyTorch;

/// <summary>An insertion-ordered dictionary produced by the unpickler (<c>dict</c> or <c>OrderedDict</c>).</summary>
internal sealed class PickleDict
{
    private readonly List<KeyValuePair<object, object?>> _items = [];
    private readonly Dictionary<object, int> _index = [];

    /// <summary>The entries in insertion order.</summary>
    public IReadOnlyList<KeyValuePair<object, object?>> Items => _items;

    /// <summary>Sets <paramref name="key"/>; an existing key keeps its position.</summary>
    public void Set(object key, object? value)
    {
        if (_index.TryGetValue(key, out int at))
        {
            _items[at] = new KeyValuePair<object, object?>(key, value);
            return;
        }

        _index[key] = _items.Count;
        _items.Add(new KeyValuePair<object, object?>(key, value));
    }

    /// <summary>The value under <paramref name="key"/>, or null.</summary>
    public object? Find(object key) => _index.TryGetValue(key, out int at) ? _items[at].Value : null;
}
