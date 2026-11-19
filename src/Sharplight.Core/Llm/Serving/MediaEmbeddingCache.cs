namespace Sharplight.Llm.Serving;

/// <summary>
/// Encoder outputs by content hash, least recently used evicted first once they exceed a byte budget. A content whose KV blocks are cached
/// needs no rows at all; this cache serves the other repeats (an image whose blocks were evicted, a prompt that differs before the image,
/// a preempted prompt computed again). Used by the scheduler thread only.
/// </summary>
internal sealed class MediaEmbeddingCache(long capacityBytes)
{
    private readonly Dictionary<UInt128, LinkedListNode<(UInt128 Hash, float[] Rows)>> _index = [];
    private readonly LinkedList<(UInt128 Hash, float[] Rows)> _lru = new();
    private long _bytes;

    /// <summary>Bytes of rows held.</summary>
    public long Bytes => _bytes;

    /// <summary>The rows of a content encoded earlier, or null (marks them most recently used).</summary>
    public float[]? Get(UInt128 hash)
    {
        if (!_index.TryGetValue(hash, out var node))
        {
            return null;
        }

        _lru.Remove(node);
        _lru.AddFirst(node);
        return node.Value.Rows;
    }

    /// <summary>Keeps the rows of a content, evicting the least recently used ones beyond the budget (rows larger than it are not kept).</summary>
    public void Add(UInt128 hash, float[] rows)
    {
        long size = (long)rows.Length * sizeof(float);
        if (size > capacityBytes || _index.ContainsKey(hash))
        {
            return;
        }

        while (_bytes + size > capacityBytes && _lru.Last is { } last)
        {
            _lru.RemoveLast();
            _index.Remove(last.Value.Hash);
            _bytes -= (long)last.Value.Rows.Length * sizeof(float);
        }

        _index[hash] = _lru.AddFirst((hash, rows));
        _bytes += size;
    }
}
