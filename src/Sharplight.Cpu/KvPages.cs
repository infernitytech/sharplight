namespace Sharplight.Cpu;

/// <summary>
/// Block table of a paged multi-sequence KV cache (<see cref="Models.InputRole.KvBlockTable"/>): position p of slot s lives in
/// cache row <c>table[s, p / BlockSize] · BlockSize + p % BlockSize</c>. A compressed cache (<c>compression</c> r &gt; 1 positions per row) keeps
/// ceil(BlockSize / r) rows per block: entry e (positions [e·r, (e + 1)·r)) is row <c>table[s, e·r / BlockSize] · ceil(BlockSize / r) + (e·r mod BlockSize) / r</c>.
/// </summary>
internal sealed unsafe class KvPages(TRef table, long blockSize, long compression = 1)
{
    /// <summary>Positions per block.</summary>
    public long BlockSize { get; } = blockSize;

    /// <summary>Positions per cache row (1, or the compression of a compressed cache).</summary>
    public long Compression { get; } = compression;

    /// <summary>Cache rows per block.</summary>
    public long RowsPerBlock { get; } = (blockSize + compression - 1) / compression;

    /// <summary>Block table columns (blocks per slot).</summary>
    public long BlocksPerSlot { get; } = table.Dims[1];

    /// <summary>Slots (block table rows).</summary>
    public long Slots { get; } = table.Dims[0];

    /// <summary>Cache row of position (compressed: entry) <paramref name="pos"/> of slot <paramref name="slot"/>.</summary>
    public long Row(int slot, long pos)
    {
        long first = pos * Compression;
        return ((int*)table.P)[slot * BlocksPerSlot + first / BlockSize] * RowsPerBlock + first % BlockSize / Compression;
    }

    /// <summary>The same block table for a compressed cache with <paramref name="positionsPerRow"/> positions per row.</summary>
    public KvPages Compressed(long positionsPerRow) => new(table, BlockSize, positionsPerRow);

    /// <summary>Throws unless the slot and position (compressed: entry) are inside the table and map to a block of a cache with <paramref name="cacheRows"/> rows.</summary>
    public void Check(int slot, long pos, long cacheRows)
    {
        long first = pos * Compression;
        if ((ulong)slot >= (ulong)Slots || pos < 0 || first >= BlocksPerSlot * BlockSize)
        {
            throw new InvalidOperationException($"Sequence slot {slot} / position {pos} outside the block table [{Slots}, {BlocksPerSlot}].");
        }

        long block = ((int*)table.P)[slot * BlocksPerSlot + first / BlockSize];
        if (block < 0 || (block + 1) * RowsPerBlock > cacheRows)
        {
            throw new InvalidOperationException($"Block {block} of slot {slot} (position {pos}) is outside the KV cache of {cacheRows} rows.");
        }
    }
}
