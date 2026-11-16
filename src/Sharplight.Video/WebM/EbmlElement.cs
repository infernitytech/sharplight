using System.Buffers.Binary;

namespace Sharplight.Video.WebM;

/// <summary>Builds EBML (Matroska / WebM) elements as byte arrays: an element id, a variable-length size and the payload.</summary>
internal static class EbmlElement
{
    /// <summary>An element containing the concatenated child elements.</summary>
    public static byte[] Master(uint id, params byte[][] children)
    {
        int length = 0;
        foreach (byte[] child in children)
        {
            length += child.Length;
        }

        var payload = new byte[length];
        int offset = 0;
        foreach (byte[] child in children)
        {
            child.CopyTo(payload, offset);
            offset += child.Length;
        }

        return Binary(id, payload);
    }

    /// <summary>An element with a raw payload.</summary>
    public static byte[] Binary(uint id, ReadOnlySpan<byte> payload)
    {
        Span<byte> idBytes = stackalloc byte[4];
        int idLength = IdBytes(id, idBytes);
        Span<byte> sizeBytes = stackalloc byte[8];
        int sizeLength = WriteSize((ulong)payload.Length, sizeBytes);
        var element = new byte[idLength + sizeLength + payload.Length];
        idBytes[..idLength].CopyTo(element);
        sizeBytes[..sizeLength].CopyTo(element.AsSpan(idLength));
        payload.CopyTo(element.AsSpan(idLength + sizeLength));
        return element;
    }

    /// <summary>An unsigned integer element in the fewest bytes.</summary>
    public static byte[] UInt(uint id, ulong value)
    {
        Span<byte> buffer = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64BigEndian(buffer, value);
        int skip = 0;
        while (skip < 7 && buffer[skip] == 0)
        {
            skip++;
        }

        return Binary(id, buffer[skip..]);
    }

    /// <summary>An 8-byte IEEE float element.</summary>
    public static byte[] Float(uint id, double value)
    {
        Span<byte> buffer = stackalloc byte[8];
        BinaryPrimitives.WriteDoubleBigEndian(buffer, value);
        return Binary(id, buffer);
    }

    /// <summary>An ASCII string element.</summary>
    public static byte[] String(uint id, string value) => Binary(id, System.Text.Encoding.ASCII.GetBytes(value));

    private static int IdBytes(uint id, Span<byte> destination)
    {
        int length = id > 0xFFFFFF ? 4 : id > 0xFFFF ? 3 : id > 0xFF ? 2 : 1;
        for (int i = 0; i < length; i++)
        {
            destination[i] = (byte)(id >> (8 * (length - 1 - i)));
        }

        return length;
    }

    private static int WriteSize(ulong size, Span<byte> destination)
    {
        int length = 1;
        while (length < 8 && size >= (1UL << (7 * length)) - 1)
        {
            length++;
        }

        ulong marked = size | (1UL << (7 * length));
        for (int i = 0; i < length; i++)
        {
            destination[i] = (byte)(marked >> (8 * (length - 1 - i)));
        }

        return length;
    }
}
