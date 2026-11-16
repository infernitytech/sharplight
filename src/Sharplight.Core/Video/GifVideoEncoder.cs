using System.Buffers.Binary;

namespace Sharplight.Video;

/// <summary>
/// Animated GIF (<see cref="VideoFormats.Gif"/>, GIF89a with the NETSCAPE looping extension). Every frame has its own local palette of up to
/// 256 colors (exact when the frame has no more than 256 distinct colors, otherwise median cut); with <see cref="VideoEncodeOptions.Quality"/>
/// below 100 quantized frames are dithered with a 4x4 ordered pattern. Frame delays are in hundredths of a second and keep the average rate.
/// </summary>
public sealed class GifVideoEncoder : IVideoEncoder
{
    /// <inheritdoc />
    public string Format => VideoFormats.Gif;

    /// <inheritdoc />
    public IReadOnlyList<string> FileExtensions { get; } = [".gif"];

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
        using var stream = new MemoryStream();
        stream.Write("GIF89a"u8);
        Span<byte> screen = stackalloc byte[7];
        BinaryPrimitives.WriteUInt16LittleEndian(screen, (ushort)clip.Width);
        BinaryPrimitives.WriteUInt16LittleEndian(screen[2..], (ushort)clip.Height);
        stream.Write(screen);
        if (options.Loop)
        {
            stream.Write([0x21, 0xFF, 0x0B]);
            stream.Write("NETSCAPE2.0"u8);
            stream.Write([0x03, 0x01, 0x00, 0x00, 0x00]);
        }

        bool dither = options.Quality < 100;
        int previousTime = 0;
        for (int i = 0; i < clip.Frames.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int time = (int)Math.Round((i + 1) * 100.0 / clip.FramesPerSecond);
            int delay = Math.Max(2, time - previousTime);
            previousTime = Math.Max(time, previousTime + 2);
            WriteFrame(stream, clip, clip.Frames[i].Rgb, delay, dither);
        }

        stream.WriteByte(0x3B);
        return stream.ToArray();
    }

    private static void WriteFrame(MemoryStream stream, VideoClip clip, byte[] rgb, int delay, bool dither)
    {
        GifPalette palette = GifPalette.Build(rgb);
        byte[] indices = palette.Map(rgb, clip.Width, dither);
        int bits = palette.BitsPerPixel;
        Span<byte> control = stackalloc byte[8];
        control[0] = 0x21;
        control[1] = 0xF9;
        control[2] = 4;
        control[3] = 0;
        BinaryPrimitives.WriteUInt16LittleEndian(control[4..], (ushort)Math.Min(delay, ushort.MaxValue));
        control[6] = 0;
        control[7] = 0;
        stream.Write(control);
        Span<byte> descriptor = stackalloc byte[10];
        descriptor[0] = 0x2C;
        BinaryPrimitives.WriteUInt16LittleEndian(descriptor[5..], (ushort)clip.Width);
        BinaryPrimitives.WriteUInt16LittleEndian(descriptor[7..], (ushort)clip.Height);
        descriptor[9] = (byte)(0x80 | (bits - 1));
        stream.Write(descriptor);
        stream.Write(palette.Table);
        var lzw = new GifLzwEncoder(bits);
        stream.WriteByte((byte)lzw.MinimumCodeSize);
        byte[] data = lzw.Encode(indices);
        for (int offset = 0; offset < data.Length; offset += 255)
        {
            int length = Math.Min(255, data.Length - offset);
            stream.WriteByte((byte)length);
            stream.Write(data, offset, length);
        }

        stream.WriteByte(0);
    }
}
