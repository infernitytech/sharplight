namespace Sharplight.Llm.Serving;

/// <summary>
/// Host-side bookkeeping of a paged KV cache: the block list of every slot (the block table the kernels read), the shared
/// <see cref="KvBlockPool"/> with its content index, and, for models with recurrent state, the snapshots of that state taken at
/// block boundaries. Prefix caching: a full block is published under the hash of its content chained from the hash of the block
/// before it (the engine computes the chain, salted with the request's cache namespace); a slot whose next blocks have the same
/// hashes attaches the cached blocks instead of computing them. A recurrent model's state cannot be rebuilt from KV rows, so its
/// slots attach a run of blocks only up to a boundary with a <em>state snapshot</em> (taken by <see cref="SaveSnapshot"/>), which
/// is restored with them. Snapshots live in <see cref="SnapshotCapacity"/> spare rows of the recurrent states, least recently used
/// first out, and are dropped together with their block when the pool evicts it.
/// Shared by <see cref="MultiSequenceDecoder"/> and test decoders; the device side is two callbacks. Not thread-safe.
/// </summary>
internal sealed class KvBlockCache
{
    private readonly KvBlockPool _pool;
    private readonly List<int>[] _blocks;
    private readonly int[] _table;
    private readonly int _blocksPerSlot;
    private readonly bool _recurrent;
    private readonly Func<int, int, bool>? _copyState;
    // Snapshots: per spare row its key, whether it is used and its last use (LRU).
    private readonly UInt128[] _snapKey;
    private readonly bool[] _snapUsed;
    private readonly long[] _snapTick;
    private readonly Dictionary<UInt128, int> _snapByKey = [];
    private long _tick;

    /// <param name="slots">Slots (rows of the block table).</param>
    /// <param name="blockSize">Tokens per block.</param>
    /// <param name="blocks">Blocks in the pool.</param>
    /// <param name="blocksPerSlot">Block table columns (the most blocks one slot holds).</param>
    /// <param name="recurrent">Slots also hold recurrent state: blocks are attached only together with a state snapshot.</param>
    /// <param name="snapshotCapacity">Spare recurrent-state rows for snapshots (rows <paramref name="slots"/> and up); 0 for none.</param>
    /// <param name="copyState">Copies all recurrent state from one row to another (slot or snapshot row); false when the device cannot.</param>
    public KvBlockCache(int slots, int blockSize, int blocks, int blocksPerSlot, bool recurrent = false, int snapshotCapacity = 0, Func<int, int, bool>? copyState = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(blockSize, 1);
        BlockSize = blockSize;
        SlotCount = slots;
        _blocksPerSlot = blocksPerSlot;
        _pool = new KvBlockPool(blocks) { Evicted = DropSnapshot };
        _blocks = [.. Enumerable.Range(0, slots).Select(_ => new List<int>())];
        _table = new int[slots * blocksPerSlot];
        Array.Fill(_table, -1);
        _recurrent = recurrent;
        _copyState = copyState;
        SnapshotCapacity = recurrent && copyState is not null ? snapshotCapacity : 0;
        _snapKey = new UInt128[SnapshotCapacity];
        _snapUsed = new bool[SnapshotCapacity];
        _snapTick = new long[SnapshotCapacity];
        TableDirty = true;
    }

    /// <summary>Tokens per block.</summary>
    public int BlockSize { get; }
    /// <summary>Slots.</summary>
    public int SlotCount { get; }
    /// <summary>Blocks in the pool.</summary>
    public int Total => _pool.Total;
    /// <summary>Blocks no slot holds, cached ones included (they are reused, least recently released first, when nothing else is free).</summary>
    public int Free => _pool.Free;
    /// <summary>Free blocks that still hold published content.</summary>
    public int Cached => _pool.Cached;
    /// <summary>Cached blocks dropped to reuse their memory.</summary>
    public long Evictions => _pool.Evictions;
    /// <summary>Spare rows for recurrent-state snapshots (0: blocks of a recurrent model are never attached).</summary>
    public int SnapshotCapacity { get; private set; }
    /// <summary>Snapshots held.</summary>
    public int Snapshots => _snapByKey.Count;
    /// <summary>Whether blocks can be attached by content (always for attention-only models; with snapshots for recurrent ones).</summary>
    public bool CachesBlocks => !_recurrent || SnapshotCapacity > 0;
    /// <summary>The block table [slots, blocks per slot] (-1 = unused).</summary>
    public int[] Table => _table;
    /// <summary>Set when <see cref="Table"/> changed; the owner clears it after uploading.</summary>
    public bool TableDirty { get; set; }

    /// <summary>Blocks of a slot, in position order.</summary>
    public IReadOnlyList<int> Blocks(int slot) => _blocks[slot];

    /// <summary>Blocks needed for <paramref name="tokens"/> tokens.</summary>
    public int BlocksFor(long tokens) => (int)((tokens + BlockSize - 1) / BlockSize);

    /// <summary>Blocks a slot of <paramref name="length"/> tokens must newly take to append <paramref name="tokens"/>.</summary>
    public int Needed(int slot, int length, int tokens) => Math.Max(0, BlocksFor((long)length + tokens) - _blocks[slot].Count);

    /// <summary>Blocks that become free when the slot is reset (those no other slot holds).</summary>
    public int Releasable(int slot)
    {
        int n = 0;
        foreach (int b in _blocks[slot])
        {
            n += _pool.RefCount(b) == 1 ? 1 : 0;
        }

        return n;
    }

    /// <summary>Takes blocks so the slot holds <paramref name="tokens"/> tokens.</summary>
    /// <exception cref="InvalidOperationException">The pool has no free block.</exception>
    public void Grow(int slot, long tokens)
    {
        var list = _blocks[slot];
        for (int j = list.Count, n = BlocksFor(tokens); j < n; j++)
        {
            int b = _pool.Allocate();
            list.Add(b);
            _table[slot * _blocksPerSlot + j] = b;
            TableDirty = true;
        }
    }

    /// <summary>Releases the slot's blocks from index <paramref name="keep"/> on, the last first (tails are evicted before prefixes).</summary>
    public void Release(int slot, int keep)
    {
        var list = _blocks[slot];
        for (int j = list.Count - 1; j >= keep; j--)
        {
            _pool.Release(list[j]);
            _table[slot * _blocksPerSlot + j] = -1;
            list.RemoveAt(j);
            TableDirty = true;
        }
    }

    /// <summary>
    /// Truncates a slot to <paramref name="length"/> tokens and returns the length kept: a cut inside a block that another slot
    /// holds or that is published (its rows are shared content) drops that whole block, so shared rows are never rewritten.
    /// </summary>
    public int Truncate(int slot, int length)
    {
        var list = _blocks[slot];
        int keep = BlocksFor(length);
        if (length % BlockSize != 0 && keep <= list.Count)
        {
            int b = list[keep - 1];
            if (_pool.RefCount(b) > 1 || _pool.IsPublished(b))
            {
                keep--;
                length = keep * BlockSize;
            }
        }

        Release(slot, keep);
        return length;
    }

    /// <summary>Maps the first <paramref name="blocks"/> blocks of <paramref name="source"/> into the empty <paramref name="slot"/>.</summary>
    public void Share(int slot, int source, int blocks)
    {
        var from = _blocks[source];
        for (int j = 0; j < blocks; j++)
        {
            _pool.Retain(from[j]);
            _blocks[slot].Add(from[j]);
            _table[slot * _blocksPerSlot + j] = from[j];
            TableDirty = true;
        }
    }

    /// <summary>
    /// How many of <paramref name="hashes"/> (the content hashes of a slot's next blocks) can be attached: the cached run from the
    /// start, for a recurrent model cut back to the last boundary with a state snapshot. <paramref name="inUse"/>: those blocks held by
    /// other slots (attaching them takes no free block).
    /// </summary>
    public int Find(ReadOnlySpan<UInt128> hashes, out int inUse)
    {
        inUse = 0;
        if (!CachesBlocks)
        {
            return 0;
        }

        int run = 0, held = 0, best = 0, bestHeld = 0;
        foreach (var h in hashes)
        {
            if (!_pool.TryFind(h, out int b))
            {
                break;
            }

            run++;
            held += _pool.RefCount(b) > 0 ? 1 : 0;
            if (!_recurrent || _snapByKey.ContainsKey(h))
            {
                best = run;
                bestHeld = held;
            }
        }

        inUse = bestHeld;
        return best;
    }

    /// <summary>
    /// Attaches the cached blocks <see cref="Find"/> reports to the slot (whose blocks must all be full) and, for a recurrent model,
    /// restores the state snapshot of the last one into the slot's row. Returns the blocks attached.
    /// </summary>
    /// <exception cref="InvalidOperationException">The slot's last block is not full (<paramref name="length"/> is not a whole number of blocks).</exception>
    public int Attach(int slot, int length, ReadOnlySpan<UInt128> hashes)
    {
        var list = _blocks[slot];
        if (length != list.Count * BlockSize)
        {
            throw new InvalidOperationException($"Slot {slot} must hold whole blocks to attach cached ones ({length} tokens, {list.Count} blocks).");
        }

        int n = Find(hashes, out _);
        if (n == 0 || list.Count + n > _blocksPerSlot)
        {
            return 0;
        }

        if (_recurrent && !RestoreSnapshot(slot, hashes[n - 1]))
        {
            return 0;
        }

        for (int j = 0; j < n; j++)
        {
            _pool.TryFind(hashes[j], out int b);
            _pool.Retain(b);
            _table[slot * _blocksPerSlot + list.Count] = b;
            list.Add(b);
        }

        TableDirty = true;
        return n;
    }

    /// <summary>Publishes the slot's full blocks <paramref name="first"/> .. under <paramref name="hashes"/>.</summary>
    public void Publish(int slot, int first, ReadOnlySpan<UInt128> hashes)
    {
        var list = _blocks[slot];
        for (int j = 0; j < hashes.Length && first + j < list.Count; j++)
        {
            _pool.Publish(list[first + j], hashes[j]);
        }
    }

    /// <summary>Whether a snapshot of recurrent state exists for the block boundary <paramref name="hash"/>.</summary>
    public bool HasSnapshot(UInt128 hash) => _snapByKey.ContainsKey(hash);

    /// <summary>
    /// Copies the slot's recurrent state (which must be the state after its first <c>k</c> blocks, the last of them published as
    /// <paramref name="hash"/>) into a snapshot row, replacing the least recently used snapshot when all rows are taken.
    /// </summary>
    public bool SaveSnapshot(int slot, UInt128 hash)
    {
        if (SnapshotCapacity == 0 || !_pool.TryFind(hash, out _))
        {
            return false;
        }

        if (_snapByKey.TryGetValue(hash, out int row))
        {
            _snapTick[row] = ++_tick;
            return true;
        }

        row = -1;
        for (int r = 0; r < SnapshotCapacity; r++)
        {
            if (!_snapUsed[r])
            {
                row = r;
                break;
            }

            if (row < 0 || _snapTick[r] < _snapTick[row])
            {
                row = r;
            }
        }

        if (_snapUsed[row])
        {
            _snapByKey.Remove(_snapKey[row]);
            _snapUsed[row] = false;
        }

        if (!_copyState!(slot, SlotCount + row))
        {
            SnapshotCapacity = 0;   // the device cannot copy state: recurrent prefix caching is off
            _snapByKey.Clear();
            return false;
        }

        _snapKey[row] = hash;
        _snapUsed[row] = true;
        _snapTick[row] = ++_tick;
        _snapByKey[hash] = row;
        return true;
    }

    private bool RestoreSnapshot(int slot, UInt128 hash)
    {
        if (!_snapByKey.TryGetValue(hash, out int row) || !_copyState!(SlotCount + row, slot))
        {
            return false;
        }

        _snapTick[row] = ++_tick;
        return true;
    }

    private void DropSnapshot(UInt128 hash)
    {
        if (_snapByKey.Remove(hash, out int row))
        {
            _snapUsed[row] = false;
        }
    }
}
