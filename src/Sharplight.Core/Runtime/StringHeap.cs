namespace Sharplight.Runtime;

/// <summary>
/// Run-owned string storage behind <see cref="Tensors.DataType.Str"/> tensors: a tensor element is the 32-bit handle <see cref="Add"/> returned.
/// Strings are interned, so equal strings share one handle and handle equality is string equality. Not thread-safe; string kernels run on one
/// thread.
/// </summary>
public sealed class StringHeap
{
    private readonly List<string> _strings = [];
    private readonly Dictionary<string, int> _handles = new(StringComparer.Ordinal);

    /// <summary>Number of distinct strings.</summary>
    public int Count => _strings.Count;

    /// <summary>The string behind a handle.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The handle was not returned by <see cref="Add"/> since the last <see cref="Clear"/>.</exception>
    public string this[int handle] => (uint)handle < (uint)_strings.Count
        ? _strings[handle]
        : throw new ArgumentOutOfRangeException(nameof(handle), $"String handle {handle} is not in the heap ({_strings.Count} strings).");

    /// <summary>Interns <paramref name="value"/> and returns its handle.</summary>
    public int Add(string value)
    {
        if (!_handles.TryGetValue(value, out int handle))
        {
            handle = _strings.Count;
            _strings.Add(value);
            _handles.Add(value, handle);
        }

        return handle;
    }

    /// <summary>Removes every string; earlier handles become invalid.</summary>
    public void Clear()
    {
        _strings.Clear();
        _handles.Clear();
    }
}
