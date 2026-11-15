namespace Sharplight.Llm.Serving;

/// <summary>
/// The physical blocks of a paged KV cache: a free list and a reference count per block (a block shared by several sequences'
/// block tables, e.g. a common prompt prefix, returns to the free list when the last of them releases it). Not thread-safe.
/// </summary>
internal sealed class KvBlockPool
{
    private readonly int[] _refs;
    private readonly Stack<int> _free;

    /// <param name="blocks">Number of blocks in the cache.</param>
    public KvBlockPool(int blocks)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(blocks, 1);
        _refs = new int[blocks];
        _free = new Stack<int>(blocks);
        for (int b = blocks - 1; b >= 0; b--)
        {
            _free.Push(b);
        }
    }

    /// <summary>Blocks in the cache.</summary>
    public int Total => _refs.Length;

    /// <summary>Blocks not held by any sequence.</summary>
    public int Free => _free.Count;

    /// <summary>Number of block tables holding <paramref name="block"/>.</summary>
    public int RefCount(int block) => _refs[block];

    /// <summary>Takes a free block (reference count 1).</summary>
    /// <exception cref="InvalidOperationException">No block is free.</exception>
    public int Allocate()
    {
        if (!_free.TryPop(out int block))
        {
            throw new InvalidOperationException("The KV cache has no free block.");
        }

        _refs[block] = 1;
        return block;
    }

    /// <summary>Adds a reference to a held block (sharing it with another sequence).</summary>
    public void Retain(int block)
    {
        if (_refs[block] <= 0)
        {
            throw new InvalidOperationException($"KV block {block} is not allocated.");
        }

        _refs[block]++;
    }

    /// <summary>Drops a reference; the block is free again when none is left.</summary>
    public void Release(int block)
    {
        if (_refs[block] <= 0)
        {
            throw new InvalidOperationException($"KV block {block} is not allocated.");
        }

        if (--_refs[block] == 0)
        {
            _free.Push(block);
        }
    }
}
