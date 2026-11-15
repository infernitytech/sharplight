using System.Buffers.Binary;
using Sharplight.Models;

namespace Sharplight.Import.Gguf;

/// <summary>
/// The tensor bytes of a GGUF tensor of type I2_S (Microsoft BitNet's llama.cpp fork, ggml id 36), read as
/// <see cref="Sharplight.Tensors.DataType.Q2S_G128"/> blocks.
/// </summary>
/// <remarks>
/// The file holds <c>n / 4</c> bytes of 2-bit codes (n elements): every 128 consecutive values take 32 bytes, byte j holding the codes of
/// values j, 32 + j, 64 + j and 96 + j in bits 6, 4, 2 and 0 (codes 0, 1, 2 for -1, 0, +1), then one f32 scale for the whole tensor
/// (the tensor is padded to 32 bytes after the codes). Each 36-byte block read here is that scale followed by the 32 code bytes.
/// </remarks>
internal sealed unsafe class GgufI2sSource : IWeightSource
{
    private const int GroupValues = 128;
    private const int CodeBytes = 32;
    private const int BlockBytes = 4 + CodeBytes;

    private readonly GgufFile _file;
    private readonly long _offset;
    private readonly Lazy<float> _scale;

    /// <param name="file">The open file.</param>
    /// <param name="offset">Absolute offset of the tensor data.</param>
    /// <param name="length">The size of the tensor as <see cref="Sharplight.Tensors.DataType.Q2S_G128"/> blocks, in bytes.</param>
    public GgufI2sSource(GgufFile file, long offset, long length)
    {
        _file = file;
        _offset = offset;
        ByteLength = length;
        _scale = new(() => BinaryPrimitives.ReadSingleLittleEndian(new ReadOnlySpan<byte>(file.BasePointer + offset + length / BlockBytes * CodeBytes, 4)));
    }

    /// <summary>The bytes the file stores for a tensor of <paramref name="blockBytes"/> bytes as blocks: the codes, the scale and the padding.</summary>
    internal static long StoredBytes(long blockBytes) => blockBytes / BlockBytes * CodeBytes + 32;

    /// <inheritdoc/>
    public long ByteLength { get; }

    /// <inheritdoc/>
    public void Read(long offset, Span<byte> destination)
    {
        if (offset < 0 || offset + destination.Length > ByteLength)
        {
            throw new ArgumentOutOfRangeException(nameof(offset));
        }

        _file.ThrowIfDisposed();
        float scale = _scale.Value;
        byte* data = _file.BasePointer + _offset;
        Span<byte> block = stackalloc byte[BlockBytes];
        long end = offset + destination.Length;
        int written = 0;
        for (long b = offset / BlockBytes; b * BlockBytes < end; b++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(block, scale);
            new ReadOnlySpan<byte>(data + b * CodeBytes, CodeBytes).CopyTo(block[4..]);
            long blockStart = b * BlockBytes;
            int from = (int)Math.Max(0, offset - blockStart);
            int to = (int)Math.Min(BlockBytes, end - blockStart);
            block[from..to].CopyTo(destination[written..]);
            written += to - from;
        }
    }

    /// <inheritdoc/>
    public nint TryGetPointer() => 0;
}
