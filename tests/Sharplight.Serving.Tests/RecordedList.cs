using System.Collections;

namespace Sharplight.Serving.Tests;

/// <summary>
/// A list that a fake device appends to on the engine's scheduler thread while a test reads it: every member takes the list's
/// lock, and enumeration (LINQ, <c>ToList</c>, <c>Assert.Contains</c>) walks a copy taken under it.
/// </summary>
public sealed class RecordedList<T> : IReadOnlyList<T>
{
    private readonly List<T> _items = [];

    /// <summary>Items recorded so far.</summary>
    public int Count
    {
        get
        {
            lock (_items)
            {
                return _items.Count;
            }
        }
    }

    /// <summary>The item recorded <paramref name="index"/>-th.</summary>
    public T this[int index]
    {
        get
        {
            lock (_items)
            {
                return _items[index];
            }
        }
    }

    /// <summary>Records <paramref name="item"/>.</summary>
    public void Add(T item)
    {
        lock (_items)
        {
            _items.Add(item);
        }
    }

    /// <summary>Forgets every item recorded so far.</summary>
    public void Clear()
    {
        lock (_items)
        {
            _items.Clear();
        }
    }

    /// <summary>Walks a copy of the items recorded so far.</summary>
    public IEnumerator<T> GetEnumerator()
    {
        T[] copy;
        lock (_items)
        {
            copy = [.. _items];
        }

        return ((IEnumerable<T>)copy).GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
