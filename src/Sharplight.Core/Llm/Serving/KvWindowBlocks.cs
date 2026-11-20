namespace Sharplight.Llm.Serving;

/// <summary>
/// The window part of a <see cref="KvBlockCache"/>: the blocks of the KV caches that only sliding-window attention reads
/// (<see cref="Graph.MultiSequence.Result.WindowStates"/>), kept in a pool of their own and only while a slot's window still reaches
/// them. Logical block j of a slot holds positions [j · block size, (j + 1) · block size) as in the main table; a query at position p
/// reads keys from p − window + 1 on, so blocks wholly before that (less a margin for truncation, e.g. rejected drafts) go back to the
/// pool. A full block is published under the same content hash as the main block at its index, so when it leaves the window it stays
/// cached (evictable) and a later prompt with the same prefix can attach it, like the main blocks. Not thread-safe.
/// </summary>
internal sealed class KvWindowBlocks
{
    private const int None = -1;
    private readonly KvBlockPool _pool;
    private readonly List<int>[] _blocks;
    private readonly int[] _trimmed;   // per slot: logical blocks before this index are known to be dropped
    private readonly int _blocksPerSlot;
    private readonly Func<int, int, int, bool>? _copyKv;

    /// <param name="slots">Slots (rows of the block table).</param>
    /// <param name="blockSize">Tokens per block.</param>
    /// <param name="blocks">Blocks in the window pool.</param>
    /// <param name="blocksPerSlot">Block table columns.</param>
    /// <param name="window">Keys each query sees (its last <paramref name="window"/> positions, itself included).</param>
    /// <param name="margin">Positions kept before the window so a slot can be truncated by this many rows (rejected drafts).</param>
    /// <param name="copyKv">Copies rows of every window cache (first source row, first destination row, rows); null when the device cannot.</param>
    public KvWindowBlocks(int slots, int blockSize, int blocks, int blocksPerSlot, int window, int margin, Func<int, int, int, bool>? copyKv)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(window, 1);
        BlockSize = blockSize;
        Window = window;
        Margin = Math.Max(0, margin);
        _blocksPerSlot = blocksPerSlot;
        _copyKv = copyKv;
        _pool = new KvBlockPool(blocks);
        _blocks = [.. Enumerable.Range(0, slots).Select(_ => new List<int>())];
        _trimmed = new int[slots];
        Table = new int[slots * blocksPerSlot];
        TableDirty = true;
    }

    /// <summary>Tokens per block.</summary>
    public int BlockSize { get; }
    /// <summary>Keys each query sees.</summary>
    public int Window { get; }
    /// <summary>Positions kept before the window (truncation margin).</summary>
    public int Margin { get; }
    /// <summary>Blocks in the window pool.</summary>
    public int Total => _pool.Total;
    /// <summary>Blocks no slot holds (cached ones included).</summary>
    public int Free => _pool.Free;
    /// <summary>The window block table [slots, blocks per slot]; entries outside a slot's window name block 0.</summary>
    public int[] Table { get; }
    /// <summary>Set when <see cref="Table"/> changed; the owner clears it after uploading.</summary>
    public bool TableDirty { get; set; }
    /// <summary>Whether blocks can be copied (copy-on-write of a shared tail, forks).</summary>
    public bool CanCopy => _copyKv is not null;

    /// <summary>
    /// Blocks a window pool needs so that it never runs out for <paramref name="slots"/> slots with steps of up to <paramref name="batchTokens"/>
    /// rows: every slot's window and margin, its partly filled block and a copy-on-write block, plus the new rows of one step.
    /// </summary>
    public static int PoolBlocks(int slots, int blockSize, int window, int margin, int batchTokens)
        => slots * ((window + margin + blockSize - 1) / blockSize + 3) + (batchTokens + blockSize - 1) / blockSize;

    /// <summary>The first logical block a query at position <paramref name="position"/> reads.</summary>
    public int FirstRead(long position) => (int)(Math.Max(0, position - Window + 1) / BlockSize);

    /// <summary>The first logical block a slot of <paramref name="length"/> tokens keeps (its next query's window, less the margin).</summary>
    private int FirstKept(long length) => (int)(Math.Max(0, length - Window + 1 - Margin) / BlockSize);

    private int BlocksFor(long tokens) => (int)((tokens + BlockSize - 1) / BlockSize);

    /// <summary>Whether the slot holds every block a query at position <paramref name="length"/> reads (so it can continue from there).</summary>
    public bool Holds(int slot, long length)
    {
        var list = _blocks[slot];
        int end = Math.Min(list.Count, BlocksFor(length));
        for (int j = FirstRead(length); j < end; j++)
        {
            if (list[j] == None)
            {
                return false;
            }
        }

        return end == BlocksFor(length);
    }

    /// <summary>Takes window blocks so the slot holds <paramref name="tokens"/> tokens.</summary>
    /// <exception cref="InvalidOperationException">The window pool has no free block.</exception>
    public void Grow(int slot, long tokens)
    {
        var list = _blocks[slot];
        for (int j = list.Count, n = BlocksFor(tokens); j < n; j++)
        {
            int b = _pool.Allocate();
            list.Add(b);
            Table[slot * _blocksPerSlot + j] = b;
            TableDirty = true;
        }
    }

    /// <summary>Returns the blocks the window of a slot of <paramref name="length"/> tokens no longer reaches (cached when published).</summary>
    public void Trim(int slot, long length)
    {
        var list = _blocks[slot];
        int end = Math.Min(list.Count, FirstKept(length));
        for (int j = _trimmed[slot]; j < end; j++)
        {
            Drop(slot, list, j);
        }

        _trimmed[slot] = Math.Max(_trimmed[slot], end);
    }

    /// <summary>Releases the slot's blocks from logical index <paramref name="keep"/> on.</summary>
    public void Release(int slot, int keep)
    {
        var list = _blocks[slot];
        for (int j = list.Count - 1; j >= keep; j--)
        {
            Drop(slot, list, j);
            list.RemoveAt(j);
        }

        _trimmed[slot] = Math.Min(_trimmed[slot], list.Count);
    }

    private void Drop(int slot, List<int> list, int j)
    {
        if (list[j] == None)
        {
            return;
        }

        _pool.Release(list[j]);
        list[j] = None;
        Table[slot * _blocksPerSlot + j] = 0;
        TableDirty = true;
    }

    /// <summary>Whether the empty <paramref name="slot"/> can share the first <paramref name="blocks"/> blocks of <paramref name="source"/> (their window is held).</summary>
    public bool CanShare(int source, int blocks) => Holds(source, (long)blocks * BlockSize) || blocks == 0;

    /// <summary>Shares the first <paramref name="blocks"/> logical blocks of <paramref name="source"/> that it holds with the empty <paramref name="slot"/>.</summary>
    public void Share(int slot, int source, int blocks)
    {
        var from = _blocks[source];
        var list = _blocks[slot];
        for (int j = 0; j < blocks; j++)
        {
            int b = j < from.Count ? from[j] : None;
            if (b != None)
            {
                _pool.Retain(b);
            }

            list.Add(b);
            Table[slot * _blocksPerSlot + j] = b == None ? 0 : b;
        }

        TableDirty = true;
    }

    /// <summary>Copy-on-write of the window block the slot appends to while another slot holds it (see <see cref="KvBlockCache.Unshare"/>).</summary>
    public bool Unshare(int slot, int length)
    {
        var list = _blocks[slot];
        int k = length / BlockSize;
        if (length % BlockSize == 0 || k >= list.Count || list[k] == None || _pool.RefCount(list[k]) <= 1)
        {
            return true;
        }

        int old = list[k];
        int copy = _pool.Allocate();
        if (_copyKv is null || !_copyKv(old * BlockSize, copy * BlockSize, length % BlockSize))
        {
            _pool.Release(copy);
            return false;
        }

        _pool.Release(old);
        list[k] = copy;
        Table[slot * _blocksPerSlot + k] = copy;
        TableDirty = true;
        return true;
    }

    /// <summary>
    /// Whether a slot that attaches cached main blocks <c>start .. start + n</c> (content hashes <paramref name="hashes"/>, from
    /// <paramref name="start"/> on) finds the window blocks its next query reads: those from <paramref name="start"/> on in the cache, the
    /// earlier ones held by <paramref name="slot"/> (assumed held for slot -1).
    /// </summary>
    public bool Reaches(int slot, int start, ReadOnlySpan<UInt128> hashes, int n)
    {
        int end = start + n;
        for (int j = FirstRead((long)end * BlockSize); j < end; j++)
        {
            if (j >= start ? !_pool.TryFind(hashes[j - start], out _) : slot >= 0 && (j >= _blocks[slot].Count || _blocks[slot][j] == None))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Attaches the cached window blocks the slot's next query reads after main blocks <c>start .. start + n</c> were attached.</summary>
    public void Attach(int slot, int start, ReadOnlySpan<UInt128> hashes, int n)
    {
        var list = _blocks[slot];
        int first = FirstRead((long)(start + n) * BlockSize);
        for (int j = start; j < start + n; j++)
        {
            int b = None;
            if (j >= first && _pool.TryFind(hashes[j - start], out b))
            {
                _pool.Retain(b);
            }
            else
            {
                b = None;
            }

            list.Add(b);
            Table[slot * _blocksPerSlot + j] = b == None ? 0 : b;
        }

        TableDirty = true;
    }

    /// <summary>Publishes the slot's held window blocks <paramref name="first"/> .. under the main blocks' hashes.</summary>
    public void Publish(int slot, int first, ReadOnlySpan<UInt128> hashes)
    {
        var list = _blocks[slot];
        for (int j = 0; j < hashes.Length && first + j < list.Count; j++)
        {
            if (list[first + j] != None)
            {
                _pool.Publish(list[first + j], hashes[j]);
            }
        }
    }
}
