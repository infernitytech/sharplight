using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Settings of <see cref="BlockSparseIndexer"/>, read from the <c>{arch}.attention.block_indexer.*</c> keys: <c>block_size</c> (key positions per
/// block, 0 = no block indexer), <c>blocks</c> (blocks every query attends) and <c>local_blocks</c> (the query's own block and the ones before
/// it that are always attended, default 1).
/// </summary>
/// <param name="BlockSize">Key positions per block.</param>
/// <param name="Blocks">Blocks every query attends.</param>
/// <param name="LocalBlocks">Latest blocks that are always attended.</param>
internal sealed record BlockSparseIndexerOptions(int BlockSize, int Blocks, int LocalBlocks)
{
    /// <summary>Key positions every query attends at most: <see cref="Blocks"/> · <see cref="BlockSize"/>.</summary>
    public int TokenBudget => Blocks * BlockSize;

    /// <summary>Reads the options of <paramref name="src"/>; null when the model has no block indexer.</summary>
    /// <exception cref="InvalidDataException">The block count is not positive.</exception>
    public static BlockSparseIndexerOptions? Read(ModelSource src)
    {
        int blockSize = src.Arch("attention.block_indexer.block_size", 0);
        if (blockSize <= 0)
        {
            return null;
        }

        int blocks = src.Arch("attention.block_indexer.blocks", 0);
        return blocks > 0
            ? new BlockSparseIndexerOptions(blockSize, blocks, src.Arch("attention.block_indexer.local_blocks", 1))
            : throw new InvalidDataException("attention.block_indexer.blocks must be positive.");
    }
}
