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
    private readonly Func<int, int, int, bool>? _copyKv;
    // Snapshots: per spare row its key, whether it is used and its last use (LRU).
    private readonly UInt128[] _snapKey;
    private readonly bool[] _snapUsed;
    private readonly long[] _snapTick;
    private readonly bool[] _snapSpeculative;
    private readonly bool[] _snapReserved;   // rows a step stores a state into before it is published (ReserveSnapshot)
    private readonly Dictionary<UInt128, int> _snapByKey = [];
    private readonly KvWindowBlocks? _window;
    private long _tick;

    /// <param name="slots">Slots (rows of the block table).</param>
    /// <param name="blockSize">Tokens per block.</param>
    /// <param name="blocks">Blocks in the pool.</param>
    /// <param name="blocksPerSlot">Block table columns (the most blocks one slot holds).</param>
    /// <param name="recurrent">Slots also hold recurrent state: blocks are attached only together with a state snapshot.</param>
    /// <param name="snapshotCapacity">Spare recurrent-state rows for snapshots (rows <paramref name="slots"/> and up); 0 for none.</param>
    /// <param name="copyState">Copies all recurrent state from one row to another (slot or snapshot row); false when the device cannot.</param>
    /// <param name="copyKv">
    /// Copies KV cache rows (first source row, first destination row, rows) of every paged cache; false when the device cannot. Needed
    /// to fork a slot (<see cref="Fork"/>): a block that two slots share and one of them writes into is copied first (copy-on-write).
    /// </param>
    /// <param name="window">
    /// The window-sized caches' blocks (<see cref="KvWindowBlocks"/>), managed alongside the main blocks: a slot attaches, shares, forks
    /// and truncates only where its window blocks are held or cached too; null when every cache has a row per position of the pool.
    /// </param>
    public KvBlockCache(int slots, int blockSize, int blocks, int blocksPerSlot, bool recurrent = false, int snapshotCapacity = 0, Func<int, int, bool>? copyState = null,
        Func<int, int, int, bool>? copyKv = null, KvWindowBlocks? window = null)
    {
        _window = window;
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
        _copyKv = copyKv;
        SnapshotCapacity = recurrent && copyState is not null ? snapshotCapacity : 0;
        _snapKey = new UInt128[SnapshotCapacity];
        _snapUsed = new bool[SnapshotCapacity];
        _snapTick = new long[SnapshotCapacity];
        _snapSpeculative = new bool[SnapshotCapacity];
        _snapReserved = new bool[SnapshotCapacity];
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
    /// <summary>The window-sized caches' blocks, or null.</summary>
    public KvWindowBlocks? Window => _window;
    /// <summary>Set when <see cref="Table"/> changed; the owner clears it after uploading.</summary>
    public bool TableDirty { get; set; }

    /// <summary>Blocks of a slot, in position order.</summary>
    public IReadOnlyList<int> Blocks(int slot) => _blocks[slot];

    /// <summary>Blocks needed for <paramref name="tokens"/> tokens.</summary>
    public int BlocksFor(long tokens) => (int)((tokens + BlockSize - 1) / BlockSize);

    /// <summary>
    /// Blocks a slot of <paramref name="length"/> tokens must newly take to append <paramref name="tokens"/>, including the copy of a
    /// partly filled last block it shares with another slot (<see cref="Unshare"/>).
    /// </summary>
    public int Needed(int slot, int length, int tokens)
        => Math.Max(0, BlocksFor((long)length + tokens) - _blocks[slot].Count) + (tokens > 0 && SharesTail(slot, length) ? 1 : 0);

    /// <summary>Whether <see cref="Fork"/> works: the device copies KV rows (and recurrent state, for a recurrent model).</summary>
    public bool CanFork => _copyKv is not null && (!_recurrent || _copyState is not null) && _window is not { CanCopy: false };

    /// <summary>
    /// Makes the empty <paramref name="slot"/> continue <paramref name="source"/>'s context: it shares all of the source's blocks,
    /// the partly filled last one included (copied by the first slot that writes into it, see <see cref="Unshare"/>), and for a
    /// recurrent model gets a copy of the source's state. Returns false, changing nothing, when the slot is not empty or the device
    /// cannot copy.
    /// </summary>
    public bool Fork(int slot, int source)
    {
        if (!CanFork || slot == source || _blocks[slot].Count != 0 || (_recurrent && !_copyState!(source, slot)))
        {
            return false;
        }

        var from = _blocks[source];
        for (int j = 0; j < from.Count; j++)
        {
            _pool.Retain(from[j]);
            _blocks[slot].Add(from[j]);
            _table[slot * _blocksPerSlot + j] = from[j];
        }

        _window?.Share(slot, source, from.Count);
        TableDirty = true;
        return true;
    }

    /// <summary>
    /// Copy-on-write: before a slot of <paramref name="length"/> tokens appends to its partly filled last block while another slot
    /// holds that block too, the block's rows are copied into a free block that replaces it in this slot. Returns false when the
    /// copy failed (the device cannot copy), leaving the slot as it was.
    /// </summary>
    /// <exception cref="InvalidOperationException">The pool has no free block.</exception>
    public bool Unshare(int slot, int length)
    {
        if (_window is not null && !_window.Unshare(slot, length))
        {
            return false;
        }

        if (!SharesTail(slot, length))
        {
            return true;
        }

        var list = _blocks[slot];
        int k = length / BlockSize;
        int old = list[k];
        int copy = _pool.Allocate();
        if (_copyKv is null || !_copyKv(old * BlockSize, copy * BlockSize, length % BlockSize))
        {
            _pool.Release(copy);
            return false;
        }

        _pool.Release(old);
        list[k] = copy;
        _table[slot * _blocksPerSlot + k] = copy;
        TableDirty = true;
        return true;
    }

    /// <summary>Whether the slot's context ends inside a block that another slot also holds.</summary>
    private bool SharesTail(int slot, int length)
    {
        var list = _blocks[slot];
        int k = length / BlockSize;
        return length % BlockSize != 0 && k < list.Count && _pool.RefCount(list[k]) > 1;
    }

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

        _window?.Grow(slot, tokens);
    }

    /// <summary>Returns the window blocks a slot of <paramref name="length"/> tokens no longer reaches (no-op without window-sized caches).</summary>
    public void Trim(int slot, long length) => _window?.Trim(slot, length);

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

        _window?.Release(slot, keep);
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

        if (_window is not null && length > 0 && !_window.Holds(slot, length))
        {
            // The window blocks before the cut are gone: the slot starts over.
            keep = 0;
            length = 0;
        }

        Release(slot, keep);
        return length;
    }

    /// <summary>
    /// Maps the first <paramref name="blocks"/> blocks of <paramref name="source"/> into the empty <paramref name="slot"/>, or fewer when
    /// the source no longer holds the window blocks after them; returns the blocks shared.
    /// </summary>
    public int Share(int slot, int source, int blocks)
    {
        while (blocks > 0 && _window is not null && !_window.CanShare(source, blocks))
        {
            blocks--;
        }

        var from = _blocks[source];
        for (int j = 0; j < blocks; j++)
        {
            _pool.Retain(from[j]);
            _blocks[slot].Add(from[j]);
            _table[slot * _blocksPerSlot + j] = from[j];
            TableDirty = true;
        }

        _window?.Share(slot, source, blocks);
        return blocks;
    }

    /// <summary>
    /// How many of <paramref name="hashes"/> (the content hashes of a slot's next blocks) can be attached: the cached run from the
    /// start, for a recurrent model cut back to the last boundary with a state snapshot. <paramref name="inUse"/>: those blocks held by
    /// other slots (attaching them takes no free block).
    /// </summary>
    public int Find(ReadOnlySpan<UInt128> hashes, out int inUse) => Find(hashes, out inUse, -1, 0);

    /// <summary>
    /// <see cref="Find(ReadOnlySpan{UInt128}, out int)"/> for <paramref name="slot"/>, whose first <paramref name="start"/> blocks precede
    /// <paramref name="hashes"/> (with window-sized caches the slot's own window blocks before them count; slot -1 assumes them held).
    /// </summary>
    private int Find(ReadOnlySpan<UInt128> hashes, out int inUse, int slot, int start)
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
            if ((!_recurrent || _snapByKey.ContainsKey(h)) && (_window is null || _window.Reaches(slot, start, hashes, run)))
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

        int start = list.Count;
        int n = Find(hashes, out _, slot, start);
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

        _window?.Attach(slot, start, hashes, n);
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

        _window?.Publish(slot, first, hashes);
    }

    /// <summary>Whether a snapshot of recurrent state exists for the block boundary <paramref name="hash"/>.</summary>
    public bool HasSnapshot(UInt128 hash) => _snapByKey.ContainsKey(hash);

    /// <summary>
    /// Copies the slot's recurrent state (which must be the state after its first <c>k</c> blocks, the last of them published as
    /// <paramref name="hash"/>) into a snapshot row, replacing the least recently used snapshot when all rows are taken. A
    /// <paramref name="speculative"/> snapshot (no prompt is known to want it) replaces only another speculative one, and is not taken when
    /// every row holds a wanted snapshot; a restored snapshot counts as wanted from then on.
    /// </summary>
    public bool SaveSnapshot(int slot, UInt128 hash, bool speculative = false)
    {
        if (SnapshotCapacity == 0 || !_pool.TryFind(hash, out _))
        {
            return false;
        }

        if (_snapByKey.TryGetValue(hash, out int row))
        {
            _snapTick[row] = ++_tick;
            _snapSpeculative[row] &= speculative;
            return true;
        }

        row = TakeRow(speculative);
        if (row < 0)
        {
            return false;
        }

        if (!_copyState!(slot, SlotCount + row))
        {
            SnapshotCapacity = 0;   // the device cannot copy state: recurrent prefix caching is off
            _snapByKey.Clear();
            return false;
        }

        _snapKey[row] = hash;
        _snapUsed[row] = true;
        _snapSpeculative[row] = speculative;
        _snapTick[row] = ++_tick;
        _snapByKey[hash] = row;
        return true;
    }

    /// <summary>
    /// Picks the snapshot row a new snapshot takes (free first, then least recently used, speculative snapshots before wanted ones; a
    /// <paramref name="speculative"/> one replaces only another speculative one) and drops its snapshot; -1 when there is none.
    /// </summary>
    private int TakeRow(bool speculative)
    {
        int row = -1;
        for (int r = 0; r < SnapshotCapacity; r++)
        {
            if (_snapReserved[r])
            {
                continue;
            }

            if (!_snapUsed[r])
            {
                row = r;
                break;
            }

            // Speculative snapshots go first (least recently used of them), then the least recently used of all.
            if (speculative && !_snapSpeculative[r])
            {
                continue;
            }

            if (row < 0 || (_snapSpeculative[r] && !_snapSpeculative[row]) || (_snapSpeculative[r] == _snapSpeculative[row] && _snapTick[r] < _snapTick[row]))
            {
                row = r;
            }
        }

        if (row >= 0 && _snapUsed[row])
        {
            _snapByKey.Remove(_snapKey[row]);
            _snapUsed[row] = false;
        }

        return row;
    }

    /// <summary>
    /// Reserves a snapshot row for a state a step stores before its block is published (in-step snapshots,
    /// <see cref="Graph.RecurrentSnapshots"/>): the row is taken as by <see cref="SaveSnapshot"/> (a wanted snapshot) and held until
    /// <see cref="CommitSnapshot"/> or <see cref="ReleaseSnapshot"/>. Returns the state row (<see cref="SlotCount"/> and up), or -1.
    /// </summary>
    public int ReserveSnapshot()
    {
        int row = SnapshotCapacity == 0 ? -1 : TakeRow(speculative: false);
        if (row < 0)
        {
            return -1;
        }

        _snapReserved[row] = true;
        return SlotCount + row;
    }

    /// <summary>
    /// Keeps the state a step stored into the reserved <paramref name="stateRow"/> as the snapshot of the published block
    /// <paramref name="hash"/>; when that block is not cached (any more) or already has a snapshot, the row is released instead (the
    /// existing snapshot is refreshed). Returns whether a snapshot for <paramref name="hash"/> exists afterwards.
    /// </summary>
    public bool CommitSnapshot(int stateRow, UInt128 hash, bool speculative = false)
    {
        int row = stateRow - SlotCount;
        if ((uint)row >= (uint)SnapshotCapacity || !_snapReserved[row])
        {
            return false;
        }

        _snapReserved[row] = false;
        if (_snapByKey.TryGetValue(hash, out int existing))
        {
            _snapTick[existing] = ++_tick;
            _snapSpeculative[existing] &= speculative;
            return true;
        }

        if (!_pool.TryFind(hash, out _))
        {
            return false;
        }

        _snapKey[row] = hash;
        _snapUsed[row] = true;
        _snapSpeculative[row] = speculative;
        _snapTick[row] = ++_tick;
        _snapByKey[hash] = row;
        return true;
    }

    /// <summary>Frees a row reserved by <see cref="ReserveSnapshot"/> that holds no wanted state.</summary>
    public void ReleaseSnapshot(int stateRow)
    {
        int row = stateRow - SlotCount;
        if ((uint)row < (uint)SnapshotCapacity)
        {
            _snapReserved[row] = false;
        }
    }

    private bool RestoreSnapshot(int slot, UInt128 hash)
    {
        if (!_snapByKey.TryGetValue(hash, out int row) || !_copyState!(SlotCount + row, slot))
        {
            return false;
        }

        _snapTick[row] = ++_tick;
        _snapSpeculative[row] = false;
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
