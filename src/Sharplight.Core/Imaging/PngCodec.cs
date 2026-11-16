using System.Buffers.Binary;
using System.IO.Compression;
using Sharplight.Vision;

namespace Sharplight.Imaging;

/// <summary>
/// PNG encoding and decoding for 8-bit RGB images (<see cref="ImageInput"/>) with the BCL's zlib, so generated images can be saved and
/// image inputs read without an imaging dependency. Encoding writes truecolor RGB. Decoding reads non-interlaced PNGs with 8 bits
/// per sample (gray, gray + alpha, RGB, RGBA, 8-bit palette) and drops the alpha channel; other files (16-bit, Adam7 interlacing,
/// low bit depths) are rejected, and other image formats are the host's job.
/// </summary>
public static class PngCodec
{
    private const byte ColorGray = 0;
    private const byte ColorRgb = 2;
    private const byte ColorPalette = 3;
    private const byte ColorGrayAlpha = 4;
    private const byte ColorRgba = 6;

    private static readonly uint[] s_crcTable = CrcTable();

    private static ReadOnlySpan<byte> Signature => [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A];

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

    /// <summary>Encodes an image with an alpha channel (a cut-out) as RGBA PNG bytes.</summary>
    /// <param name="image">The image.</param>
    /// <param name="alpha">The opacity of every pixel, <c>Width · Height</c> bytes (255 opaque).</param>
    /// <param name="level">The zlib compression level.</param>
    /// <exception cref="ArgumentException">The image or the alpha plane has the wrong size.</exception>
    public static byte[] EncodeRgba(ImageInput image, ReadOnlySpan<byte> alpha, CompressionLevel level = CompressionLevel.Optimal)
    {
        ArgumentNullException.ThrowIfNull(image);
        image.Validate();
        if (alpha.Length != image.Width * image.Height)
        {
            throw new ArgumentException($"Expected {image.Width * image.Height} alpha bytes, got {alpha.Length}.", nameof(alpha));
        }

        using var stream = new MemoryStream();
        stream.Write(Signature);
        var header = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(header, image.Width);
        BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), image.Height);
        header[8] = 8;
        header[9] = ColorRgba;
        WriteChunk(stream, "IHDR"u8, header);
        using var compressed = new MemoryStream();
        using (var zlib = new ZLibStream(compressed, level, leaveOpen: true))
        {
            var row = new byte[(image.Width * 4) + 1];
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    int i = (y * image.Width) + x;
                    row[1 + (4 * x)] = image.Rgb[3 * i];
                    row[2 + (4 * x)] = image.Rgb[(3 * i) + 1];
                    row[3 + (4 * x)] = image.Rgb[(3 * i) + 2];
                    row[4 + (4 * x)] = alpha[i];
                }

                zlib.Write(row);
            }
        }

        WriteChunk(stream, "IDAT"u8, compressed.ToArray());
        WriteChunk(stream, "IEND"u8, []);
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
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(image);
        image.Validate();
        stream.Write(Signature);
        var header = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(header, image.Width);
        BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), image.Height);
        header[8] = 8;
        header[9] = ColorRgb;
        WriteChunk(stream, "IHDR"u8, header);
        WriteChunk(stream, "IDAT"u8, CompressRows(image, level));
        WriteChunk(stream, "IEND"u8, []);
    }

    /// <summary>Decodes PNG bytes.</summary>
    /// <param name="png">The file contents.</param>
    /// <exception cref="InvalidDataException">The data is not a PNG this decoder supports, or it is damaged (bad CRC, truncated).</exception>
    public static ImageInput Decode(ReadOnlySpan<byte> png)
    {
        if (png.Length < Signature.Length || !png[..Signature.Length].SequenceEqual(Signature))
        {
            throw new InvalidDataException("Not a PNG file.");
        }

        var header = default(PngHeader);
        byte[]? palette = null;
        using var compressed = new MemoryStream();
        int position = Signature.Length;
        while (true)
        {
            if (position + 12 > png.Length)
            {
                throw new InvalidDataException("PNG file is truncated.");
            }

            int length = BinaryPrimitives.ReadInt32BigEndian(png[position..]);
            if (length < 0 || position + 12L + length > png.Length)
            {
                throw new InvalidDataException("PNG chunk is truncated.");
            }

            var type = png.Slice(position + 4, 4);
            var data = png.Slice(position + 8, length);
            uint crc = BinaryPrimitives.ReadUInt32BigEndian(png[(position + 8 + length)..]);
            if ((Crc(Crc(0xFFFFFFFFu, type), data) ^ 0xFFFFFFFFu) != crc)
            {
                throw new InvalidDataException($"PNG chunk {System.Text.Encoding.ASCII.GetString(type)} has a bad CRC.");
            }

            position += 12 + length;
            if (type.SequenceEqual("IHDR"u8))
            {
                header = PngHeader.Read(data);
            }
            else if (type.SequenceEqual("PLTE"u8))
            {
                palette = data.ToArray();
            }
            else if (type.SequenceEqual("IDAT"u8))
            {
                compressed.Write(data);
            }
            else if (type.SequenceEqual("IEND"u8))
            {
                break;
            }
        }

        if (header.Width == 0)
        {
            throw new InvalidDataException("PNG file has no IHDR chunk.");
        }

        compressed.Position = 0;
        var samples = Unfilter(compressed, header);
        return ToRgb(samples, header, palette);
    }

    /// <summary>Reads a PNG file.</summary>
    /// <param name="path">The file.</param>
    /// <exception cref="InvalidDataException">The file is not a PNG this decoder supports, or it is damaged.</exception>
    public static ImageInput Read(string path) => Decode(File.ReadAllBytes(path));

    internal static byte[] CompressRows(ImageInput image, CompressionLevel level)
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

    /// <summary>Inflates the image data and reverses the per-row filters; returns the raw samples, rows without filter bytes.</summary>
    private static byte[] Unfilter(Stream compressed, PngHeader header)
    {
        int bytesPerPixel = header.Channels;
        int stride = checked(header.Width * bytesPerPixel);
        var samples = new byte[checked(stride * header.Height)];
        var row = new byte[stride + 1];
        using var zlib = new ZLibStream(compressed, CompressionMode.Decompress);
        for (int y = 0; y < header.Height; y++)
        {
            try
            {
                zlib.ReadExactly(row);
            }
            catch (EndOfStreamException error)
            {
                throw new InvalidDataException("PNG image data is truncated.", error);
            }

            var current = samples.AsSpan(y * stride, stride);
            var previous = y == 0 ? Span<byte>.Empty : samples.AsSpan((y - 1) * stride, stride);
            row.AsSpan(1).CopyTo(current);
            Reconstruct(row[0], current, previous, bytesPerPixel);
        }

        return samples;
    }

    private static void Reconstruct(byte filter, Span<byte> current, ReadOnlySpan<byte> previous, int bytesPerPixel)
    {
        for (int i = 0; i < current.Length; i++)
        {
            int left = i >= bytesPerPixel ? current[i - bytesPerPixel] : 0;
            int up = previous.IsEmpty ? 0 : previous[i];
            int upLeft = i >= bytesPerPixel && !previous.IsEmpty ? previous[i - bytesPerPixel] : 0;
            int predictor = filter switch
            {
                0 => 0,
                1 => left,
                2 => up,
                3 => (left + up) / 2,
                4 => Paeth(left, up, upLeft),
                _ => throw new InvalidDataException($"Unknown PNG filter type {filter}."),
            };
            current[i] = (byte)(current[i] + predictor);
        }
    }

    private static int Paeth(int left, int up, int upLeft)
    {
        int estimate = left + up - upLeft;
        int distanceLeft = Math.Abs(estimate - left);
        int distanceUp = Math.Abs(estimate - up);
        int distanceUpLeft = Math.Abs(estimate - upLeft);
        if (distanceLeft <= distanceUp && distanceLeft <= distanceUpLeft)
        {
            return left;
        }

        return distanceUp <= distanceUpLeft ? up : upLeft;
    }

    private static ImageInput ToRgb(byte[] samples, PngHeader header, byte[]? palette)
    {
        int pixels = header.Width * header.Height;
        var rgb = new byte[pixels * 3];
        for (int i = 0; i < pixels; i++)
        {
            int source = i * header.Channels;
            int target = i * 3;
            switch (header.ColorType)
            {
                case ColorGray:
                case ColorGrayAlpha:
                    rgb[target] = rgb[target + 1] = rgb[target + 2] = samples[source];
                    break;
                case ColorPalette:
                    int entry = samples[source] * 3;
                    if (palette is null || entry + 2 >= palette.Length)
                    {
                        throw new InvalidDataException("PNG palette index outside the palette.");
                    }

                    rgb[target] = palette[entry];
                    rgb[target + 1] = palette[entry + 1];
                    rgb[target + 2] = palette[entry + 2];
                    break;
                default:
                    rgb[target] = samples[source];
                    rgb[target + 1] = samples[source + 1];
                    rgb[target + 2] = samples[source + 2];
                    break;
            }
        }

        return new ImageInput(header.Width, header.Height, rgb);
    }

    internal static void WriteChunk(Stream stream, ReadOnlySpan<byte> type, ReadOnlySpan<byte> data)
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

    /// <summary>The IHDR fields the decoder uses.</summary>
    private readonly record struct PngHeader(int Width, int Height, byte ColorType, int Channels)
    {
        public static PngHeader Read(ReadOnlySpan<byte> data)
        {
            if (data.Length < 13)
            {
                throw new InvalidDataException("PNG IHDR chunk is too short.");
            }

            int width = BinaryPrimitives.ReadInt32BigEndian(data);
            int height = BinaryPrimitives.ReadInt32BigEndian(data[4..]);
            byte bitDepth = data[8];
            byte colorType = data[9];
            byte interlace = data[12];
            if (width <= 0 || height <= 0)
            {
                throw new InvalidDataException($"PNG image size {width}x{height} has no pixels.");
            }

            if (bitDepth != 8 || interlace != 0)
            {
                throw new InvalidDataException($"Unsupported PNG: bit depth {bitDepth}, interlace {interlace} (8-bit non-interlaced only).");
            }

            int channels = colorType switch
            {
                ColorGray or ColorPalette => 1,
                ColorGrayAlpha => 2,
                ColorRgb => 3,
                ColorRgba => 4,
                _ => throw new InvalidDataException($"Unsupported PNG color type {colorType}."),
            };
            return new PngHeader(width, height, colorType, channels);
        }
    }
}
