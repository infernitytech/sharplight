using System.Buffers.Binary;
using System.IO.Compression;

namespace Sharplight.Video.Tests;

/// <summary>A small independent APNG decoder for the tests (filter type 0 only, which is what the encoder writes; blend source, dispose none).</summary>
internal static class ApngReader
{
    public static (int Width, int Height, int PlayCount, List<(byte[] Rgb, int DelayNumerator, int DelayDenominator)> Frames) Read(byte[] file)
    {
        Assert.True(file.AsSpan(0, 8).SequenceEqual(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }));
        int position = 8;
        int width = 0;
        int height = 0;
        int frameCount = -1;
        int playCount = -1;
        int sequence = 0;
        var frames = new List<(byte[] Rgb, int N, int D)>();
        byte[]? canvas = null;
        (int X, int Y, int W, int H, int N, int D) control = default;
        var data = new MemoryStream();
        bool haveControl = false;

        void Finish()
        {
            if (!haveControl)
            {
                return;
            }

            data.Position = 0;
            using var zlib = new ZLibStream(data, CompressionMode.Decompress);
            var row = new byte[control.W * 3 + 1];
            for (int y = 0; y < control.H; y++)
            {
                zlib.ReadExactly(row);
                Assert.Equal(0, row[0]);
                Array.Copy(row, 1, canvas!, ((control.Y + y) * width + control.X) * 3, control.W * 3);
            }

            frames.Add(((byte[])canvas!.Clone(), control.N, control.D));
            data = new MemoryStream();
            haveControl = false;
        }

        while (position < file.Length)
        {
            int length = BinaryPrimitives.ReadInt32BigEndian(file.AsSpan(position));
            string type = System.Text.Encoding.ASCII.GetString(file, position + 4, 4);
            ReadOnlySpan<byte> body = file.AsSpan(position + 8, length);
            switch (type)
            {
                case "IHDR":
                    width = BinaryPrimitives.ReadInt32BigEndian(body);
                    height = BinaryPrimitives.ReadInt32BigEndian(body[4..]);
                    canvas = new byte[width * height * 3];
                    break;
                case "acTL":
                    frameCount = BinaryPrimitives.ReadInt32BigEndian(body);
                    playCount = BinaryPrimitives.ReadInt32BigEndian(body[4..]);
                    break;
                case "fcTL":
                    Finish();
                    Assert.Equal(sequence++, BinaryPrimitives.ReadInt32BigEndian(body));
                    control = (
                        BinaryPrimitives.ReadInt32BigEndian(body[12..]),
                        BinaryPrimitives.ReadInt32BigEndian(body[16..]),
                        BinaryPrimitives.ReadInt32BigEndian(body[4..]),
                        BinaryPrimitives.ReadInt32BigEndian(body[8..]),
                        BinaryPrimitives.ReadUInt16BigEndian(body[20..]),
                        BinaryPrimitives.ReadUInt16BigEndian(body[22..]));
                    haveControl = true;
                    break;
                case "IDAT":
                    data.Write(body);
                    break;
                case "fdAT":
                    Assert.Equal(sequence++, BinaryPrimitives.ReadInt32BigEndian(body));
                    data.Write(body[4..]);
                    break;
                case "IEND":
                    Finish();
                    break;
            }

            position += 12 + length;
        }

        Assert.Equal(frameCount, frames.Count);
        return (width, height, playCount, frames.Select(f => (f.Rgb, f.N, f.D)).ToList());
    }
}
