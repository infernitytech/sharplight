using System.Buffers.Binary;

namespace Sharplight.Import.PyTorch;

/// <summary>A read-only block of memory (typically a memory-mapped file, possibly larger than 2 GB) addressed with 64-bit offsets and bounds-checked.</summary>
internal readonly unsafe struct FileView(byte* start, long length)
{
    /// <summary>Size of the block in bytes.</summary>
    public long Length => length;

    /// <summary>The bytes <paramref name="offset"/> to <paramref name="offset"/> + <paramref name="count"/>.</summary>
    /// <exception cref="InvalidDataException">The range is outside the block.</exception>
    public ReadOnlySpan<byte> Slice(long offset, int count)
    {
        if (offset < 0 || count < 0 || offset > length - count)
        {
            throw new InvalidDataException("A structure of the checkpoint lies outside the file (truncated file?).");
        }

        return new ReadOnlySpan<byte>(start + offset, count);
    }

    /// <summary>Reads a little-endian 16-bit value.</summary>
    public ushort U16(long offset) => BinaryPrimitives.ReadUInt16LittleEndian(Slice(offset, 2));

    /// <summary>Reads a little-endian 32-bit value.</summary>
    public uint U32(long offset) => BinaryPrimitives.ReadUInt32LittleEndian(Slice(offset, 4));

    /// <summary>Reads a little-endian 64-bit value that must fit a signed long.</summary>
    public long I64(long offset)
    {
        long value = BinaryPrimitives.ReadInt64LittleEndian(Slice(offset, 8));
        return value >= 0 ? value : throw new InvalidDataException("A 64-bit field of the checkpoint is out of range.");
    }
}
