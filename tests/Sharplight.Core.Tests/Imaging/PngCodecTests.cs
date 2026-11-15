using System.Buffers.Binary;
using System.IO.Compression;
using Sharplight.Imaging;
using Sharplight.Vision;
using Xunit;

namespace Sharplight.Core.Tests.Imaging;

public class PngCodecTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(7, 5)]
    [InlineData(64, 33)]
    public void EncodeDecodeRoundTrip(int width, int height)
    {
        var image = RandomImage(width, height, seed: width * 31 + height);
        var png = PngCodec.Encode(image);
        var decoded = PngCodec.Decode(png);
        Assert.Equal((width, height), (decoded.Width, decoded.Height));
        Assert.Equal(image.Rgb, decoded.Rgb);
    }

    [Fact]
    public void FileRoundTrip()
    {
        var path = Path.Combine(Path.GetTempPath(), $"sharplight-png-{Guid.NewGuid():N}.png");
        try
        {
            var image = RandomImage(10, 4, seed: 3);
            PngCodec.Write(path, image);
            Assert.Equal(image.Rgb, PngCodec.Read(path).Rgb);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void DecodesEveryRowFilter(byte filter)
    {
        var image = RandomImage(9, 6, seed: filter);
        var png = TestPng.Build(image.Width, image.Height, colorType: 2, image.Rgb, filter);
        Assert.Equal(image.Rgb, PngCodec.Decode(png).Rgb);
    }

    [Fact]
    public void DecodesGrayAlphaRgbaAndPalette()
    {
        byte[] gray = [0, 128, 255, 7];
        Assert.Equal(new byte[] { 0, 0, 0, 128, 128, 128, 255, 255, 255, 7, 7, 7 }, PngCodec.Decode(TestPng.Build(2, 2, 0, gray, 4)).Rgb);

        byte[] grayAlpha = [10, 255, 20, 0];
        Assert.Equal(new byte[] { 10, 10, 10, 20, 20, 20 }, PngCodec.Decode(TestPng.Build(2, 1, 4, grayAlpha, 1)).Rgb);

        byte[] rgba = [1, 2, 3, 255, 4, 5, 6, 0];
        Assert.Equal(new byte[] { 1, 2, 3, 4, 5, 6 }, PngCodec.Decode(TestPng.Build(1, 2, 6, rgba, 2)).Rgb);

        byte[] indices = [1, 0, 1];
        byte[] palette = [9, 8, 7, 1, 2, 3];
        Assert.Equal(new byte[] { 1, 2, 3, 9, 8, 7, 1, 2, 3 }, PngCodec.Decode(TestPng.Build(3, 1, 3, indices, 0, palette)).Rgb);
    }

    [Fact]
    public void RejectsDamagedAndUnsupportedFiles()
    {
        var png = PngCodec.Encode(RandomImage(4, 4, seed: 1));
        var damaged = (byte[])png.Clone();
        damaged[^20] ^= 0xFF;
        Assert.Throws<InvalidDataException>(() => PngCodec.Decode(damaged));
        Assert.Throws<InvalidDataException>(() => PngCodec.Decode(png.AsSpan(0, png.Length - 30)));
        Assert.Throws<InvalidDataException>(() => PngCodec.Decode("GIF89a.........."u8));
        var sixteenBit = TestPng.Build(1, 1, 2, new byte[6], 0, bitDepth: 16);
        Assert.Throws<InvalidDataException>(() => PngCodec.Decode(sixteenBit));
    }

    [Fact]
    public void EncodedFileStartsWithSignatureAndHeader()
    {
        var png = PngCodec.Encode(RandomImage(3, 2, seed: 9));
        Assert.Equal(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }, png[..8]);
        Assert.Equal("IHDR"u8.ToArray(), png[12..16]);
        Assert.Equal(3, BinaryPrimitives.ReadInt32BigEndian(png.AsSpan(16)));
        Assert.Equal(2, BinaryPrimitives.ReadInt32BigEndian(png.AsSpan(20)));
    }

    internal static ImageInput RandomImage(int width, int height, int seed)
    {
        var rgb = new byte[width * height * 3];
        new Random(seed).NextBytes(rgb);
        return new ImageInput(width, height, rgb);
    }

    /// <summary>Writes PNGs the codec does not produce itself: other color types and row filters.</summary>
    private static class TestPng
    {
        public static byte[] Build(int width, int height, byte colorType, byte[] samples, byte filter, byte[]? palette = null, byte bitDepth = 8)
        {
            int channels = colorType switch
            {
                0 or 3 => 1,
                4 => 2,
                2 => 3,
                _ => 4,
            };
            int stride = width * channels * (bitDepth / 8);
            using var stream = new MemoryStream();
            stream.Write([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
            var header = new byte[13];
            BinaryPrimitives.WriteInt32BigEndian(header, width);
            BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), height);
            header[8] = bitDepth;
            header[9] = colorType;
            Chunk(stream, "IHDR"u8, header);
            if (palette is not null)
            {
                Chunk(stream, "PLTE"u8, palette);
            }

            using var compressed = new MemoryStream();
            using (var zlib = new ZLibStream(compressed, CompressionLevel.Fastest, leaveOpen: true))
            {
                for (int y = 0; y < height; y++)
                {
                    var row = samples.AsSpan(y * stride, stride);
                    var previous = y == 0 ? ReadOnlySpan<byte>.Empty : samples.AsSpan((y - 1) * stride, stride);
                    zlib.WriteByte(filter);
                    zlib.Write(Filter(filter, row, previous, channels));
                }
            }

            Chunk(stream, "IDAT"u8, compressed.ToArray());
            Chunk(stream, "IEND"u8, []);
            return stream.ToArray();
        }

        private static byte[] Filter(byte filter, ReadOnlySpan<byte> row, ReadOnlySpan<byte> previous, int bpp)
        {
            var output = new byte[row.Length];
            for (int i = 0; i < row.Length; i++)
            {
                int left = i >= bpp ? row[i - bpp] : 0;
                int up = previous.IsEmpty ? 0 : previous[i];
                int upLeft = i >= bpp && !previous.IsEmpty ? previous[i - bpp] : 0;
                int predictor = filter switch
                {
                    1 => left,
                    2 => up,
                    3 => (left + up) / 2,
                    4 => Paeth(left, up, upLeft),
                    _ => 0,
                };
                output[i] = (byte)(row[i] - predictor);
            }

            return output;
        }

        private static int Paeth(int a, int b, int c)
        {
            int p = a + b - c;
            int pa = Math.Abs(p - a);
            int pb = Math.Abs(p - b);
            int pc = Math.Abs(p - c);
            if (pa <= pb && pa <= pc)
            {
                return a;
            }

            return pb <= pc ? b : c;
        }

        private static void Chunk(Stream stream, ReadOnlySpan<byte> type, ReadOnlySpan<byte> data)
        {
            Span<byte> word = stackalloc byte[4];
            BinaryPrimitives.WriteInt32BigEndian(word, data.Length);
            stream.Write(word);
            stream.Write(type);
            stream.Write(data);
            BinaryPrimitives.WriteUInt32BigEndian(word, Crc32([.. type.ToArray(), .. data.ToArray()]));
            stream.Write(word);
        }

        private static uint Crc32(byte[] bytes)
        {
            uint crc = 0xFFFFFFFFu;
            foreach (byte b in bytes)
            {
                crc ^= b;
                for (int k = 0; k < 8; k++)
                {
                    crc = (crc & 1) != 0 ? 0xEDB88320u ^ (crc >> 1) : crc >> 1;
                }
            }

            return crc ^ 0xFFFFFFFFu;
        }
    }
}
