namespace Sharplight.Shaders.Lowering;


/// <summary>First-fit allocator for the activation arena (offsets aligned to 256 bytes).</summary>
internal sealed class ActivationAllocator
{
    private readonly SortedList<long, long> _free = [];   // offset → size
    private readonly Dictionary<object, (long Off, long Size)> _live = [];
    private long _top, _high;
    public long HighWater => _high;

    public long Allocate(object v, long bytes)
    {
        bytes = (bytes + 255) & ~255L;
        foreach (var (off, size) in _free)
        {
            if (size < bytes)
            {
                continue;
            }

            _free.Remove(off);
            if (size > bytes)
            {
                _free.Add(off + bytes, size - bytes);
            }

            _live[v] = (off, bytes);
            return off;
        }
        long o = _top;
        _top += bytes;
        _high = Math.Max(_high, _top);
        _live[v] = (o, bytes);
        return o;
    }

    public void Free(object v)
    {
        if (!_live.Remove(v, out var a))
        {
            return;
        }

        _free.Add(a.Off, a.Size);
        // coalesce neighbours
        bool merged = true;
        while (merged)
        {
            merged = false;
            for (int i = 0; i + 1 < _free.Count; i++)
            {
                if (_free.Keys[i] + _free.Values[i] == _free.Keys[i + 1])
                {
                    long off = _free.Keys[i];
                    long size = _free.Values[i] + _free.Values[i + 1];
                    _free.RemoveAt(i + 1);
                    _free[off] = size;
                    merged = true;
                    break;
                }
            }
        }
        // shrink the top
        if (_free.Count > 0 && _free.Keys[^1] + _free.Values[^1] == _top)
        {
            _top = _free.Keys[^1];
            _free.RemoveAt(_free.Count - 1);
        }
    }
}
