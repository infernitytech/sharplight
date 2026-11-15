using System.Buffers.Binary;
using System.Text;

namespace Sharplight.Studio.Services.Imaging;

/// <summary>
/// Text metadata of PNG files (<c>tEXt</c> chunks with Latin-1 keywords and values as the PNG specification has them; a value with characters
/// beyond Latin-1 goes into an uncompressed UTF-8 <c>iTXt</c> chunk). Generated images carry their prompt and parameters this way, so a picture
/// tells how it was made; programs that read PNG text chunks (and Studio) show it.
/// </summary>
internal static class PngMetadata
{
    private static readonly byte[] s_signature = [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A];
    private static readonly uint[] s_crcTable = CrcTable();

    /// <summary>Returns <paramref name="png"/> with the entries added as text chunks before the image data.</summary>
    /// <param name="png">A PNG file's bytes.</param>
    /// <param name="entries">Keyword and text pairs (keywords are 1 to 79 characters).</param>
    /// <exception cref="InvalidDataException">The bytes are not a PNG.</exception>
    public static byte[] Embed(byte[] png, IEnumerable<KeyValuePair<string, string>> entries)
    {
        if (png.Length < 33 || !png.AsSpan(0, 8).SequenceEqual(s_signature))
        {
            throw new InvalidDataException("Not a PNG file.");
        }

        // The signature and the IHDR chunk (8 + 4 + 4 + 13 + 4 bytes) come first; the text goes right after it.
        const int AfterHeader = 33;
        using var output = new MemoryStream(png.Length + 256);
        output.Write(png, 0, AfterHeader);
        foreach (var (key, value) in entries)
        {
            WriteText(output, key, value);
        }

        output.Write(png, AfterHeader, png.Length - AfterHeader);
        return output.ToArray();
    }

    /// <summary>Reads the text chunks (<c>tEXt</c> and uncompressed <c>iTXt</c>) of a PNG file.</summary>
    /// <param name="png">A PNG file's bytes.</param>
    /// <returns>The entries in file order; empty for a file that is not a PNG.</returns>
    public static Dictionary<string, string> Read(ReadOnlySpan<byte> png)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        if (png.Length < 8 || !png[..8].SequenceEqual(s_signature))
        {
            return result;
        }

        int position = 8;
        while (position + 12 <= png.Length)
        {
            int length = BinaryPrimitives.ReadInt32BigEndian(png[position..]);
            var type = png.Slice(position + 4, 4);
            if (length < 0 || position + 12L + length > png.Length)
            {
                break;
            }

            var data = png.Slice(position + 8, length);
            int zero = data.IndexOf((byte)0);
            if (type.SequenceEqual("tEXt"u8) && zero > 0)
            {
                result[Encoding.Latin1.GetString(data[..zero])] = Encoding.Latin1.GetString(data[(zero + 1)..]);
            }
            else if (type.SequenceEqual("iTXt"u8) && zero > 0 && data.Length > zero + 3 && data[zero + 1] == 0)
            {
                // keyword, 0, compression flag (0), method, language tag, 0, translated keyword, 0, text.
                var rest = data[(zero + 3)..];
                int tag = rest.IndexOf((byte)0);
                if (tag >= 0)
                {
                    rest = rest[(tag + 1)..];
                    int translated = rest.IndexOf((byte)0);
                    if (translated >= 0)
                    {
                        result[Encoding.Latin1.GetString(data[..zero])] = Encoding.UTF8.GetString(rest[(translated + 1)..]);
                    }
                }
            }
            else if (type.SequenceEqual("IEND"u8))
            {
                break;
            }

            position += 12 + length;
        }

        return result;
    }

    private static void WriteText(Stream stream, string key, string value)
    {
        string keyword = key.Length > 79 ? key[..79] : key;
        using var data = new MemoryStream();
        data.Write(Encoding.Latin1.GetBytes(keyword));
        data.WriteByte(0);
        if (value.All(character => character < 256))
        {
            data.Write(Encoding.Latin1.GetBytes(value));
            WriteChunk(stream, "tEXt"u8, data.ToArray());
            return;
        }

        data.Write([0, 0, 0, 0]);
        data.Write(Encoding.UTF8.GetBytes(value));
        WriteChunk(stream, "iTXt"u8, data.ToArray());
    }

    private static void WriteChunk(Stream stream, ReadOnlySpan<byte> type, ReadOnlySpan<byte> data)
    {
        Span<byte> number = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(number, data.Length);
        stream.Write(number);
        stream.Write(type);
        stream.Write(data);
        BinaryPrimitives.WriteUInt32BigEndian(number, Crc(Crc(0xFFFFFFFFu, type), data) ^ 0xFFFFFFFFu);
        stream.Write(number);
    }

    private static uint Crc(uint crc, ReadOnlySpan<byte> bytes)
    {
        foreach (byte value in bytes)
        {
            crc = s_crcTable[(crc ^ value) & 0xFF] ^ (crc >> 8);
        }

        return crc;
    }

    private static uint[] CrcTable()
    {
        var table = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
            {
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            }

            table[n] = c;
        }

        return table;
    }
}
