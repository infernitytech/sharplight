using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// The tensor bytes of a GGUF tensor of one of the narrow IQ1 types (ggml ids 64-66, see <see cref="GgmlNarrowIq1"/>), read as IQ1_S
/// blocks: every stored block is widened to the IQ1_S block with the same values when it is read.
/// </summary>
internal sealed unsafe class GgufNarrowIq1Source : IWeightSource
{
    private readonly GgufFile _file;
    private readonly long _offset;
    private readonly int _ggmlType;
    private readonly int _storedBlockBytes;

    /// <param name="file">The open file.</param>
    /// <param name="offset">Absolute offset of the tensor data.</param>
    /// <param name="storedLength">Tensor size in the file, in bytes (whole stored blocks).</param>
    /// <param name="ggmlType">The narrow type id.</param>
    public GgufNarrowIq1Source(GgufFile file, long offset, long storedLength, int ggmlType)
    {
        _file = file;
        _offset = offset;
        _ggmlType = ggmlType;
        _storedBlockBytes = GgmlNarrowIq1.StoredBlockBytes(ggmlType);
        ByteLength = storedLength / _storedBlockBytes * GgmlNarrowIq1.WideBlockBytes;
    }

    /// <inheritdoc/>
    public long ByteLength { get; }

    /// <inheritdoc/>
    public void Read(long off, Span<byte> destination)
    {
        if (off < 0 || off + destination.Length > ByteLength)
        {
            throw new ArgumentOutOfRangeException(nameof(off));
        }

        _file.ThrowIfDisposed();
        byte* data = _file.BasePointer + _offset;
        Span<byte> block = stackalloc byte[GgmlNarrowIq1.WideBlockBytes];
        long end = off + destination.Length;
        int written = 0;
        for (long b = off / GgmlNarrowIq1.WideBlockBytes; b * GgmlNarrowIq1.WideBlockBytes < end; b++)
        {
            GgmlNarrowIq1.Widen(_ggmlType, new ReadOnlySpan<byte>(data + b * _storedBlockBytes, _storedBlockBytes), block);
            long blockStart = b * GgmlNarrowIq1.WideBlockBytes;
            int from = (int)Math.Max(0, off - blockStart);
            int to = (int)Math.Min(GgmlNarrowIq1.WideBlockBytes, end - blockStart);
            block[from..to].CopyTo(destination[written..]);
            written += to - from;
        }
    }

    /// <inheritdoc/>
    public nint TryGetPointer() => 0;
}
