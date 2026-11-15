using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// The tensor bytes of a GGUF tensor in one of the interleaved Q4_0 layouts of older llama.cpp releases (<c>Q4_0_4_4</c>,
/// <c>Q4_0_4_8</c>, <c>Q4_0_8_8</c>, ggml types 31-33, written for the ARM repacked kernels and removed from llama.cpp since), read
/// as plain Q4_0 rows. Each group of <c>R</c> consecutive rows is stored block column by block column: the <c>R</c> f16 scales of
/// the column's blocks, then their 16-byte nibble arrays cut into <c>chunk</c>-byte pieces and interleaved row by row, every byte
/// XORed with 0x88 (signed nibbles). Reading undoes that permutation; the values are exactly those of the Q4_0 blocks.
/// </summary>
internal sealed unsafe class GgufInterleavedQ4_0Source : IWeightSource
{
    private const int BlockBytes = 18;
    private const int ScaleBytes = 2;
    private const int NibbleBytes = 16;

    private readonly GgufFile _file;
    private readonly long _offset;
    private readonly long _blocksPerRow;
    private readonly int _rows;
    private readonly int _chunk;

    /// <param name="file">The open file.</param>
    /// <param name="offset">Absolute offset of the tensor data.</param>
    /// <param name="length">Tensor size in bytes (the same as the Q4_0 size).</param>
    /// <param name="blocksPerRow">32-value blocks per row.</param>
    /// <param name="interleavedRows">Rows interleaved per group (4 or 8).</param>
    /// <param name="chunk">Bytes of a nibble array kept together (4 or 8).</param>
    public GgufInterleavedQ4_0Source(GgufFile file, long offset, long length, long blocksPerRow, int interleavedRows, int chunk)
    {
        _file = file;
        _offset = offset;
        ByteLength = length;
        _blocksPerRow = blocksPerRow;
        _rows = interleavedRows;
        _chunk = chunk;
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
        Span<byte> block = stackalloc byte[BlockBytes];
        long end = off + destination.Length;
        int written = 0;
        for (long b = off / BlockBytes; b * BlockBytes < end; b++)
        {
            long row = b / _blocksPerRow;
            long column = b % _blocksPerRow;
            int within = (int)(row % _rows);
            // The interleaved block of R rows for this column: R scales, then R * 16 nibble bytes.
            byte* group = data + ((row / _rows) * _blocksPerRow + column) * _rows * BlockBytes;
            block[0] = group[within * ScaleBytes];
            block[1] = group[within * ScaleBytes + 1];
            byte* nibbles = group + _rows * ScaleBytes;
            for (int piece = 0; piece < NibbleBytes / _chunk; piece++)
            {
                byte* source = nibbles + (piece * _rows + within) * _chunk;
                for (int i = 0; i < _chunk; i++)
                {
                    block[ScaleBytes + piece * _chunk + i] = (byte)(source[i] ^ 0x88);
                }
            }

            long blockStart = b * BlockBytes;
            int from = (int)Math.Max(0, off - blockStart);
            int to = (int)Math.Min(BlockBytes, end - blockStart);
            block[from..to].CopyTo(destination[written..]);
            written += to - from;
        }
    }
}
