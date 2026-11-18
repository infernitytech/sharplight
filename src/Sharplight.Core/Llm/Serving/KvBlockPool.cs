namespace Sharplight.Llm.Serving;

/// <summary>
/// The physical blocks of a paged KV cache with a reference count per block and a content index for prefix caching. A block no
/// sequence holds is free; a free block that still carries a content hash (see <see cref="Publish"/>) is <em>cached</em>: it
/// keeps its rows and can be taken back by <see cref="Retain"/> until <see cref="Allocate"/> reuses it. Free blocks form one
/// queue ordered for eviction: blocks without content first, then cached blocks least recently released first, so releasing a
/// sequence's blocks from its last to its first (as <see cref="KvBlockCache"/> does) evicts its tail before its prefix.
/// Not thread-safe.
/// </summary>
internal sealed class KvBlockPool
{
    private const int None = -1;
    private readonly int[] _refs;
    private readonly int[] _prev, _next;   // the free queue (doubly linked), head = evicted first
    private readonly bool[] _hashed;
    private readonly UInt128[] _hash;
    private readonly Dictionary<UInt128, int> _byHash = [];
    private int _head = None, _tail = None, _free, _cached;

    /// <param name="blocks">Number of blocks in the cache.</param>
    public KvBlockPool(int blocks)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(blocks, 1);
        _refs = new int[blocks];
        _prev = new int[blocks];
        _next = new int[blocks];
        _hashed = new bool[blocks];
        _hash = new UInt128[blocks];
        for (int b = 0; b < blocks; b++)
        {
            Append(b);
        }
    }

    /// <summary>Called with the content hash of a cached block when <see cref="Allocate"/> reuses it.</summary>
    public Action<UInt128>? Evicted { get; set; }

    /// <summary>Blocks in the cache.</summary>
    public int Total => _refs.Length;

    /// <summary>Blocks not held by any sequence (including cached ones).</summary>
    public int Free => _free;

    /// <summary>Free blocks that still hold content found by <see cref="TryFind"/>.</summary>
    public int Cached => _cached;

    /// <summary>Blocks, held or cached, that carry a content hash.</summary>
    public int Indexed => _byHash.Count;

    /// <summary>Cached blocks whose content was dropped to reuse the block.</summary>
    public long Evictions { get; private set; }

    /// <summary>Number of block tables holding <paramref name="block"/>.</summary>
    public int RefCount(int block) => _refs[block];

    /// <summary>Whether <paramref name="block"/> carries a content hash (its full rows must never be rewritten).</summary>
    public bool IsPublished(int block) => _hashed[block];

    /// <summary>The block holding the content <paramref name="hash"/>, held or cached.</summary>
    public bool TryFind(UInt128 hash, out int block) => _byHash.TryGetValue(hash, out block);

    /// <summary>Takes a free block (reference count 1): one without content first, else the least recently released cached one.</summary>
    /// <exception cref="InvalidOperationException">No block is free.</exception>
    public int Allocate()
    {
        int block = _head;
        if (block == None)
        {
            throw new InvalidOperationException("The KV cache has no free block.");
        }

        Unlink(block);
        if (_hashed[block])
        {
            var h = _hash[block];
            _cached--;
            Unpublish(block);
            Evictions++;
            Evicted?.Invoke(h);
        }

        _refs[block] = 1;
        return block;
    }

    /// <summary>Adds a reference to a held block (sharing it with another sequence) or takes back a cached one.</summary>
    /// <exception cref="InvalidOperationException">The block is free and holds no content.</exception>
    public void Retain(int block)
    {
        if (_refs[block] == 0)
        {
            if (!_hashed[block])
            {
                throw new InvalidOperationException($"KV block {block} is not allocated.");
            }

            Unlink(block);
            _cached--;
        }

        _refs[block]++;
    }

    /// <summary>Drops a reference; when none is left the block becomes free (cached, at the back of the queue, if it holds content).</summary>
    /// <exception cref="InvalidOperationException">The block is not held.</exception>
    public void Release(int block)
    {
        if (_refs[block] <= 0)
        {
            throw new InvalidOperationException($"KV block {block} is not allocated.");
        }

        if (--_refs[block] > 0)
        {
            return;
        }

        if (_hashed[block])
        {
            Append(block);
            _cached++;
        }
        else
        {
            Prepend(block);
        }
    }

    /// <summary>
    /// Names a held, full block by its content hash so later sequences with the same content can share it. A hash already naming
    /// another block keeps that block (the duplicate stays private); a block already named keeps its name.
    /// </summary>
    /// <exception cref="InvalidOperationException">The block is not held.</exception>
    public void Publish(int block, UInt128 hash)
    {
        if (_refs[block] <= 0)
        {
            throw new InvalidOperationException($"KV block {block} is not allocated.");
        }

        if (_hashed[block] || !_byHash.TryAdd(hash, block))
        {
            return;
        }

        _hashed[block] = true;
        _hash[block] = hash;
    }

    /// <summary>Drops a held block's content hash (its rows are about to change).</summary>
    public void Unpublish(int block)
    {
        if (!_hashed[block])
        {
            return;
        }

        _byHash.Remove(_hash[block]);
        _hashed[block] = false;
    }

    private void Append(int block)
    {
        _prev[block] = _tail;
        _next[block] = None;
        if (_tail == None)
        {
            _head = block;
        }
        else
        {
            _next[_tail] = block;
        }

        _tail = block;
        _free++;
    }

    private void Prepend(int block)
    {
        _next[block] = _head;
        _prev[block] = None;
        if (_head == None)
        {
            _tail = block;
        }
        else
        {
            _prev[_head] = block;
        }

        _head = block;
        _free++;
    }

    private void Unlink(int block)
    {
        int p = _prev[block], n = _next[block];
        if (p == None)
        {
            _head = n;
        }
        else
        {
            _next[p] = n;
        }

        if (n == None)
        {
            _tail = p;
        }
        else
        {
            _prev[n] = p;
        }

        _free--;
    }
}
