using System.Buffers.Binary;
using System.Text;

namespace Sharplight.Distributed.Protocol;

/// <summary>Reads a frame payload written by <see cref="PayloadWriter"/>; a read past the end throws <see cref="InvalidDataException"/>.</summary>
internal sealed class PayloadReader(ReadOnlyMemory<byte> payload)
{
    private int _position;

    /// <summary>Whether every byte was read.</summary>
    public bool AtEnd => _position == payload.Length;

    public byte Byte() => Take(1).Span[0];

    public bool Bool() => Byte() != 0;

    public int Int32() => BinaryPrimitives.ReadInt32LittleEndian(Take(4).Span);

    public uint UInt32() => BinaryPrimitives.ReadUInt32LittleEndian(Take(4).Span);

    public long Int64() => BinaryPrimitives.ReadInt64LittleEndian(Take(8).Span);

    public double Double() => BinaryPrimitives.ReadDoubleLittleEndian(Take(8).Span);

    public string String() => Encoding.UTF8.GetString(Take(Count()).Span);

    /// <summary>A length-prefixed byte block (a view into the payload).</summary>
    public ReadOnlyMemory<byte> Bytes() => Take(Count());

    /// <summary>The rest of the payload.</summary>
    public ReadOnlyMemory<byte> Rest() => Take(payload.Length - _position);

    public void Map(IDictionary<string, long> destination)
    {
        int count = Count();
        for (int i = 0; i < count; i++)
        {
            string key = String();
            destination[key] = Int64();
        }
    }

    public List<string> Strings()
    {
        int count = Count();
        var values = new List<string>(Math.Min(count, 1024));
        for (int i = 0; i < count; i++)
        {
            values.Add(String());
        }

        return values;
    }

    /// <summary>A non-negative count or length.</summary>
    public int Count()
    {
        int count = Int32();
        return count >= 0 && count <= payload.Length - _position + (1 << 20) ? count : throw new InvalidDataException($"Corrupt segment message (count {count}).");
    }

    private ReadOnlyMemory<byte> Take(int length)
    {
        if (length < 0 || _position + length > payload.Length)
        {
            throw new InvalidDataException("Truncated segment message.");
        }

        var slice = payload.Slice(_position, length);
        _position += length;
        return slice;
    }
}
