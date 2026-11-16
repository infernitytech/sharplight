using System.Buffers.Binary;
using System.IO.Compression;
using Sharplight.Imaging;
using Sharplight.Vision;

namespace Sharplight.Video;

/// <summary>
/// Animated PNG (<see cref="VideoFormats.Apng"/>): lossless truecolor frames in <c>acTL</c> / <c>fcTL</c> / <c>fdAT</c> chunks. After the first
/// frame only the changed rectangle of each frame is stored. The file is a valid PNG, so viewers without APNG support show the first frame.
/// </summary>
public sealed class ApngVideoEncoder : IVideoEncoder
{
    /// <inheritdoc />
    public string Format => VideoFormats.Apng;

    /// <inheritdoc />
    public IReadOnlyList<string> FileExtensions { get; } = [".apng", ".png"];

    /// <inheritdoc />
    public bool IsAvailable => true;

    /// <inheritdoc />
    public async Task EncodeAsync(VideoClip clip, Stream output, VideoEncodeOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(clip);
        ArgumentNullException.ThrowIfNull(output);
        VideoEncodeOptions settings = options ?? new VideoEncodeOptions();
        byte[] bytes = await Task.Run(() => Encode(clip, settings, cancellationToken), cancellationToken).ConfigureAwait(false);
        await output.WriteAsync(bytes, cancellationToken).ConfigureAwait(false);
    }

    private static byte[] Encode(VideoClip clip, VideoEncodeOptions options, CancellationToken cancellationToken)
    {
        CompressionLevel level = options.Quality < 25 ? CompressionLevel.Fastest : CompressionLevel.Optimal;
        using var stream = new MemoryStream();
        stream.Write(PngSignature);
        Span<byte> header = stackalloc byte[13];
        BinaryPrimitives.WriteInt32BigEndian(header, clip.Width);
        BinaryPrimitives.WriteInt32BigEndian(header[4..], clip.Height);
        header[8] = 8;
        header[9] = 2;
        PngCodec.WriteChunk(stream, "IHDR"u8, header);
        Span<byte> control = stackalloc byte[8];
        BinaryPrimitives.WriteInt32BigEndian(control, clip.Frames.Count);
        BinaryPrimitives.WriteInt32BigEndian(control[4..], options.Loop ? 0 : 1);
        PngCodec.WriteChunk(stream, "acTL"u8, control);

        int sequence = 0;
        ImageInput? previous = null;
        for (int i = 0; i < clip.Frames.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ImageInput frame = clip.Frames[i];
            (int x, int y, int width, int height) = previous is null ? (0, 0, frame.Width, frame.Height) : ChangedRegion(previous, frame);
            ImageInput region = x == 0 && y == 0 && width == frame.Width && height == frame.Height ? frame : Crop(frame, x, y, width, height);
            WriteFrameControl(stream, sequence++, region, x, y, clip.FramesPerSecond);
            byte[] compressed = PngCodec.CompressRows(region, level);
            if (i == 0)
            {
                PngCodec.WriteChunk(stream, "IDAT"u8, compressed);
            }
            else
            {
                var data = new byte[compressed.Length + 4];
                BinaryPrimitives.WriteInt32BigEndian(data, sequence++);
                compressed.CopyTo(data, 4);
                PngCodec.WriteChunk(stream, "fdAT"u8, data);
            }

            previous = frame;
        }

        PngCodec.WriteChunk(stream, "IEND"u8, []);
        return stream.ToArray();
    }

    private static ReadOnlySpan<byte> PngSignature => [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A];

    private static void WriteFrameControl(Stream stream, int sequence, ImageInput region, int x, int y, double framesPerSecond)
    {
        // The delay is the fraction numerator / denominator seconds: exact (1 / fps) for whole rates, milliseconds otherwise.
        bool whole = framesPerSecond == Math.Floor(framesPerSecond) && framesPerSecond <= ushort.MaxValue;
        int denominator = whole ? (int)framesPerSecond : 1000;
        int numerator = whole ? 1 : (int)Math.Clamp(Math.Round(denominator / framesPerSecond), 1, ushort.MaxValue);
        Span<byte> data = stackalloc byte[26];
        BinaryPrimitives.WriteInt32BigEndian(data, sequence);
        BinaryPrimitives.WriteInt32BigEndian(data[4..], region.Width);
        BinaryPrimitives.WriteInt32BigEndian(data[8..], region.Height);
        BinaryPrimitives.WriteInt32BigEndian(data[12..], x);
        BinaryPrimitives.WriteInt32BigEndian(data[16..], y);
        BinaryPrimitives.WriteUInt16BigEndian(data[20..], (ushort)numerator);
        BinaryPrimitives.WriteUInt16BigEndian(data[22..], (ushort)denominator);
        data[24] = 0;
        data[25] = 0;
        PngCodec.WriteChunk(stream, "fcTL"u8, data);
    }

    /// <summary>The smallest rectangle containing every pixel that differs from the previous frame (one pixel when nothing changed).</summary>
    private static (int X, int Y, int Width, int Height) ChangedRegion(ImageInput previous, ImageInput current)
    {
        int width = current.Width;
        int height = current.Height;
        int minX = width;
        int minY = height;
        int maxX = -1;
        int maxY = -1;
        for (int y = 0; y < height; y++)
        {
            ReadOnlySpan<byte> before = previous.Rgb.AsSpan(y * width * 3, width * 3);
            ReadOnlySpan<byte> after = current.Rgb.AsSpan(y * width * 3, width * 3);
            if (before.SequenceEqual(after))
            {
                continue;
            }

            int first = before.CommonPrefixLength(after) / 3;
            int last = width - 1;
            while (before.Slice(last * 3, 3).SequenceEqual(after.Slice(last * 3, 3)))
            {
                last--;
            }

            minX = Math.Min(minX, first);
            maxX = Math.Max(maxX, last);
            minY = Math.Min(minY, y);
            maxY = y;
        }

        return maxX < 0 ? (0, 0, 1, 1) : (minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    private static ImageInput Crop(ImageInput image, int x, int y, int width, int height)
    {
        var rgb = new byte[width * height * 3];
        for (int row = 0; row < height; row++)
        {
            image.Rgb.AsSpan(((y + row) * image.Width + x) * 3, width * 3).CopyTo(rgb.AsSpan(row * width * 3));
        }

        return new ImageInput(width, height, rgb);
    }
}
