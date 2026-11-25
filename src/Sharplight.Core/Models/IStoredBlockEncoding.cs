namespace Sharplight.Models;

/// <summary>
/// A block-for-block re-encoding between the bytes a file stores and the bytes of the weight's data type: every stored block of
/// <see cref="StoredBlockBytes"/> becomes one block of <see cref="BlockBytes"/> with the same values (for example a narrower variant of a
/// quantization format widened to the format the kernels run). Sources that store such data in one contiguous file range expose it through
/// <see cref="IWeightSource.TryGetEncodedFileRange"/>, so that a reader with its own file I/O (routed experts on disk) can read the stored bytes
/// and decode them itself.
/// </summary>
public interface IStoredBlockEncoding
{
    /// <summary>Bytes of one stored block.</summary>
    public int StoredBlockBytes { get; }

    /// <summary>Bytes of the same block in the weight's data type.</summary>
    public int BlockBytes { get; }

    /// <summary>Decodes whole blocks: <paramref name="stored"/> holds n · <see cref="StoredBlockBytes"/> bytes, <paramref name="blocks"/> n · <see cref="BlockBytes"/>.</summary>
    /// <param name="stored">The stored blocks.</param>
    /// <param name="blocks">The blocks to write.</param>
    public void Decode(ReadOnlySpan<byte> stored, Span<byte> blocks);
}
