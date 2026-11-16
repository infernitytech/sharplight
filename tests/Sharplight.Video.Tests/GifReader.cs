using System.Buffers.Binary;

namespace Sharplight.Video.Tests;

/// <summary>A small independent GIF decoder for the tests (local palettes, full-size frames, LZW).</summary>
internal static class GifReader
{
    public static (int Width, int Height, int LoopCount, List<(byte[] Rgb, int DelayCentiseconds)> Frames) Read(byte[] file)
    {
        Assert.Equal("GIF89a", System.Text.Encoding.ASCII.GetString(file, 0, 6));
        int width = BinaryPrimitives.ReadUInt16LittleEndian(file.AsSpan(6));
        int height = BinaryPrimitives.ReadUInt16LittleEndian(file.AsSpan(8));
        byte screenFlags = file[10];
        int position = 13;
        byte[]? globalTable = null;
        if ((screenFlags & 0x80) != 0)
        {
            int size = 3 << ((screenFlags & 7) + 1);
            globalTable = file[position..(position + size)];
            position += size;
        }

        int loop = -1;
        int delay = 0;
        var frames = new List<(byte[], int)>();
        while (true)
        {
            byte marker = file[position++];
            if (marker == 0x3B)
            {
                break;
            }

            if (marker == 0x21)
            {
                byte label = file[position++];
                int blockStart = position;
                var content = new List<byte>();
                while (file[position] != 0)
                {
                    content.AddRange(file.AsSpan(position + 1, file[position]).ToArray());
                    position += file[position] + 1;
                }

                position++;
                if (label == 0xF9)
                {
                    delay = BinaryPrimitives.ReadUInt16LittleEndian(file.AsSpan(blockStart + 2));
                }
                else if (label == 0xFF && System.Text.Encoding.ASCII.GetString(content.ToArray(), 0, 11) == "NETSCAPE2.0")
                {
                    loop = BinaryPrimitives.ReadUInt16LittleEndian(content.ToArray().AsSpan(12));
                }

                continue;
            }

            Assert.Equal(0x2C, marker);
            int w = BinaryPrimitives.ReadUInt16LittleEndian(file.AsSpan(position + 4));
            int h = BinaryPrimitives.ReadUInt16LittleEndian(file.AsSpan(position + 6));
            byte flags = file[position + 8];
            position += 9;
            byte[] table = globalTable!;
            if ((flags & 0x80) != 0)
            {
                int size = 3 << ((flags & 7) + 1);
                table = file[position..(position + size)];
                position += size;
            }

            int minimum = file[position++];
            var compressed = new List<byte>();
            while (file[position] != 0)
            {
                compressed.AddRange(file.AsSpan(position + 1, file[position]).ToArray());
                position += file[position] + 1;
            }

            position++;
            byte[] indices = Lzw(compressed.ToArray(), minimum, w * h);
            var rgb = new byte[w * h * 3];
            for (int i = 0; i < indices.Length; i++)
            {
                Array.Copy(table, indices[i] * 3, rgb, i * 3, 3);
            }

            Assert.Equal(width, w);
            Assert.Equal(height, h);
            frames.Add((rgb, delay));
        }

        return (width, height, loop, frames);
    }

    private static byte[] Lzw(byte[] data, int minimumCodeSize, int pixelCount)
    {
        int clear = 1 << minimumCodeSize;
        int end = clear + 1;
        var prefix = new int[4096];
        var suffix = new byte[4096];
        var length = new int[4096];
        for (int i = 0; i < clear; i++)
        {
            suffix[i] = (byte)i;
            length[i] = 1;
        }

        var output = new byte[pixelCount];
        int outputPosition = 0;
        int codeSize = minimumCodeSize + 1;
        int next = end + 1;
        int previous = -1;
        int bitBuffer = 0;
        int bitCount = 0;
        int index = 0;
        var scratch = new byte[4096];
        while (outputPosition < pixelCount)
        {
            while (bitCount < codeSize)
            {
                bitBuffer |= data[index++] << bitCount;
                bitCount += 8;
            }

            int code = bitBuffer & ((1 << codeSize) - 1);
            bitBuffer >>= codeSize;
            bitCount -= codeSize;
            if (code == clear)
            {
                codeSize = minimumCodeSize + 1;
                next = end + 1;
                previous = -1;
                continue;
            }

            if (code == end)
            {
                break;
            }

            int current = code;
            int extra = -1;
            if (previous >= 0 && code >= next)
            {
                Assert.Equal(next, code);
                current = previous;
                extra = 1;
            }

            int count = length[current];
            int walk = current;
            for (int i = count - 1; i >= 0; i--)
            {
                scratch[i] = suffix[walk];
                walk = prefix[walk];
            }

            int first = scratch[0];
            for (int i = 0; i < count; i++)
            {
                output[outputPosition++] = scratch[i];
            }

            if (extra > 0)
            {
                output[outputPosition++] = (byte)first;
            }

            if (previous >= 0 && next < 4096)
            {
                prefix[next] = previous;
                suffix[next] = (byte)first;
                length[next] = length[previous] + 1;
                next++;
                if (next == (1 << codeSize) && codeSize < 12)
                {
                    codeSize++;
                }
            }

            previous = code;
        }

        Assert.Equal(pixelCount, outputPosition);
        return output;
    }
}
