using System.Buffers.Binary;
using System.IO.Compression;
using Sharplight.Vision;

namespace Sharplight.Imaging;

/// <summary>
/// Encodes 8-bit RGB images (<see cref="ImageInput"/>) as PNG with the BCL's zlib, so generated images can be saved without an
/// imaging dependency. Decoding (PNG, JPEG, ...) stays the host's job, as for vision inputs.
/// </summary>
public static class PngEncoder
{
    private static readonly uint[] s_crcTable = CrcTable();

    /// <summary>Encodes an image as PNG bytes.</summary>
    /// <param name="image">The image (validated: positive size, <c>3 · Width · Height</c> bytes).</param>
    /// <param name="level">The zlib compression level.</param>
    /// <exception cref="ArgumentException">The image has no pixels or the wrong number of bytes.</exception>
    public static byte[] Encode(ImageInput image, CompressionLevel level = CompressionLevel.Optimal)
    {
        using var stream = new MemoryStream();
        Write(stream, image, level);
        return stream.ToArray();
    }

    /// <summary>Writes an image as a PNG file.</summary>
    /// <param name="path">The file (created or overwritten).</param>
    /// <param name="image">The image.</param>
    /// <param name="level">The zlib compression level.</param>
    /// <exception cref="ArgumentException">The image has no pixels or the wrong number of bytes.</exception>
    public static void Write(string path, ImageInput image, CompressionLevel level = CompressionLevel.Optimal)
    {
        using var stream = File.Create(path);
        Write(stream, image, level);
    }

    /// <summary>Writes an image as PNG to a stream.</summary>
    /// <param name="stream">The stream; not disposed.</param>
    /// <param name="image">The image.</param>
    /// <param name="level">The zlib compression level.</param>
    /// <exception cref="ArgumentException">The image has no pixels or the wrong number of bytes.</exception>
    public static void Write(Stream stream, ImageInput image, CompressionLevel level = CompressionLevel.Optimal)
    {
        image.Validate();
        stream.Write([0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        var header = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(header, image.Width);
        BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), image.Height);
        header[8] = 8; // bit depth
        header[9] = 2; // color type: truecolor RGB
        WriteChunk(stream, "IHDR"u8, header);
        WriteChunk(stream, "IDAT"u8, CompressRows(image, level));
        WriteChunk(stream, "IEND"u8, []);
    }

    private static byte[] CompressRows(ImageInput image, CompressionLevel level)
    {
        int stride = image.Width * 3;
        using var compressed = new MemoryStream();
        using (var zlib = new ZLibStream(compressed, level, leaveOpen: true))
        {
            Span<byte> filter = [0];
            for (int y = 0; y < image.Height; y++)
            {
                zlib.Write(filter);
                zlib.Write(image.Rgb.AsSpan(y * stride, stride));
            }
        }

        return compressed.ToArray();
    }

    private static void WriteChunk(Stream stream, ReadOnlySpan<byte> type, ReadOnlySpan<byte> data)
    {
        Span<byte> word = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(word, data.Length);
        stream.Write(word);
        stream.Write(type);
        stream.Write(data);
        uint crc = Crc(Crc(0xFFFFFFFFu, type), data) ^ 0xFFFFFFFFu;
        BinaryPrimitives.WriteUInt32BigEndian(word, crc);
        stream.Write(word);
    }

    private static uint Crc(uint crc, ReadOnlySpan<byte> bytes)
    {
        foreach (byte b in bytes)
        {
            crc = s_crcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
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
