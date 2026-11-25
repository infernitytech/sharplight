namespace Sharplight.Models;

/// <summary>Provides the raw bytes of a weight tensor (e.g. a memory-mapped file region).</summary>
public interface IWeightSource
{
    /// <summary>Size of the stored data in bytes.</summary>
    public long ByteLength { get; }

    /// <summary>Copies <paramref name="destination"/>.Length bytes starting at <paramref name="offset"/>.</summary>
    public void Read(long offset, Span<byte> destination);

    /// <summary>
    /// Returns a pointer to the data if it is directly addressable (memory mapped), enabling zero-copy
    /// uploads; otherwise <see cref="IntPtr.Zero"/>.
    /// </summary>
    public nint TryGetPointer() => 0;

    /// <summary>
    /// The memory mapping the data lies in (the start and length of the mapped file), when the source is a view of one, so that a
    /// provider can expose the whole mapping to a device once (unified memory: one buffer over the mapped file) and address each
    /// weight at its offset. The mapping stays valid while the source is. Default: none.
    /// </summary>
    /// <param name="mappingBase">The first byte of the mapping.</param>
    /// <param name="mappingLength">Bytes in the mapping.</param>
    /// <returns><see langword="true"/> when the data lies in a mapping.</returns>
    public bool TryGetMapping(out nint mappingBase, out long mappingLength)
    {
        mappingBase = 0;
        mappingLength = 0;
        return false;
    }

    /// <summary>
    /// The file the data is stored in and the byte offset of its first byte there, when it is one contiguous range of a file (memory-mapped
    /// GGUF, <c>.slm</c> and safetensors weights), so that a consumer can read parts of it with its own file I/O instead of through the
    /// mapping (routed experts left on disk, <see cref="Runtime.Experts.ExpertStore"/>). Default: none.
    /// </summary>
    /// <param name="path">The file.</param>
    /// <param name="offset">Offset of the data's first byte in the file.</param>
    /// <returns><see langword="true"/> when the data is a contiguous range of <paramref name="path"/>.</returns>
    public bool TryGetFileRange(out string path, out long offset)
    {
        path = "";
        offset = 0;
        return false;
    }

    /// <summary>
    /// The file range of data stored in another block encoding than the weight's data type (<see cref="IStoredBlockEncoding"/>): the stored
    /// blocks are one contiguous range of <paramref name="path"/> from <paramref name="offset"/>, and <see cref="Read"/> returns them decoded.
    /// For sources whose bytes are stored as they are read, <see cref="TryGetFileRange"/> applies instead. Default: none.
    /// </summary>
    /// <param name="path">The file.</param>
    /// <param name="offset">Offset of the first stored byte in the file.</param>
    /// <param name="encoding">Decodes the stored blocks to the weight's data type.</param>
    /// <returns><see langword="true"/> when the data is such a range.</returns>
    public bool TryGetEncodedFileRange(out string path, out long offset, out IStoredBlockEncoding? encoding)
    {
        path = "";
        offset = 0;
        encoding = null;
        return false;
    }

    /// <summary>
    /// Computes the data now if this source produces it lazily on first <see cref="Read"/> (default: nothing to do), so that a
    /// consumer can produce several weights in parallel ahead of reading them. Thread-safe; concurrent reads wait for the result.
    /// </summary>
    public void Materialize()
    {
    }

    /// <summary>
    /// Drops data this source computed lazily (default: nothing to do); a later <see cref="Read"/> computes it again. Writers that
    /// stream weights into a package call it after copying a weight, so a conversion holds one derived weight at a time.
    /// </summary>
    public void Release()
    {
    }
}
