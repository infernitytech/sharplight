namespace Sharplight.Cpu;

/// <summary>
/// Block table of a paged multi-sequence KV cache (<see cref="Models.InputRole.KvBlockTable"/>): position p of slot s lives in
/// cache row <c>table[s, p / BlockSize] · BlockSize + p % BlockSize</c>.
/// </summary>
internal sealed unsafe class KvPages(TRef table, long blockSize)
{
    /// <summary>Rows per block.</summary>
    public long BlockSize { get; } = blockSize;

    /// <summary>Block table columns (blocks per slot).</summary>
    public long BlocksPerSlot { get; } = table.Dims[1];

    /// <summary>Slots (block table rows).</summary>
    public long Slots { get; } = table.Dims[0];

    /// <summary>Cache row of position <paramref name="pos"/> of slot <paramref name="slot"/>.</summary>
    public long Row(int slot, long pos) => ((int*)table.P)[slot * BlocksPerSlot + pos / BlockSize] * BlockSize + pos % BlockSize;

    /// <summary>Throws unless the slot and position are inside the table and map to a block of a cache with <paramref name="cacheRows"/> rows.</summary>
    public void Check(int slot, long pos, long cacheRows)
    {
        if ((ulong)slot >= (ulong)Slots || pos < 0 || pos >= BlocksPerSlot * BlockSize)
        {
            throw new InvalidOperationException($"Sequence slot {slot} / position {pos} outside the block table [{Slots}, {BlocksPerSlot}].");
        }

        long block = ((int*)table.P)[slot * BlocksPerSlot + pos / BlockSize];
        if (block < 0 || (block + 1) * BlockSize > cacheRows)
        {
            throw new InvalidOperationException($"Block {block} of slot {slot} (position {pos}) is outside the KV cache of {cacheRows} rows.");
        }
    }
}
