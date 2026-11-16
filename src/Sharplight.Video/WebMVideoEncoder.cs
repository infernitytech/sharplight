using Microsoft.Extensions.Options;
using Sharplight.Video.Native;
using Sharplight.Video.WebM;

namespace Sharplight.Video;

/// <summary>
/// WebM (<see cref="VideoFormats.WebM"/>) with VP9 or VP8 video, encoded by the bundled libvpx (<c>native/sharplight_video</c>) and muxed in
/// managed code. Frames are converted from RGB to 4:2:0 with BT.601 limited-range coefficients. <see cref="IsAvailable"/> is false when the
/// native library does not exist for this platform.
/// </summary>
public sealed class WebMVideoEncoder : IVideoEncoder
{
    private readonly WebMCodec _codec;

    /// <summary>Creates the encoder.</summary>
    /// <param name="options">The codec selection; <see langword="null"/> uses VP9.</param>
    public WebMVideoEncoder(IOptions<SharplightVideoOptions>? options = null)
    {
        _codec = options?.Value.WebMCodec ?? WebMCodec.Vp9;
    }

    /// <inheritdoc />
    public string Format => VideoFormats.WebM;

    /// <inheritdoc />
    public IReadOnlyList<string> FileExtensions { get; } = [".webm"];

    /// <inheritdoc />
    public bool IsAvailable => NativeVideoLibrary.IsAvailable;

    /// <inheritdoc />
    public async Task EncodeAsync(VideoClip clip, Stream output, VideoEncodeOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(clip);
        ArgumentNullException.ThrowIfNull(output);
        NativeVideoLibrary.ThrowIfUnavailable();
        VideoEncodeOptions settings = options ?? new VideoEncodeOptions();
        byte[] file = await Task.Run(() => Encode(clip, settings, cancellationToken), cancellationToken).ConfigureAwait(false);
        await output.WriteAsync(file, cancellationToken).ConfigureAwait(false);
    }

    private byte[] Encode(VideoClip clip, VideoEncodeOptions options, CancellationToken cancellationToken)
    {
        (int numerator, int denominator) = FrameRate.ToFraction(clip.FramesPerSecond);
        int threads = options.MaxThreads > 0 ? options.MaxThreads : Math.Clamp(Environment.ProcessorCount, 1, 8);
        int bitrate = options.BitrateKilobitsPerSecond ?? 0;
        using var encoder = new NativeVpxEncoder(_codec, clip.Width, clip.Height, numerator, denominator, bitrate, Math.Clamp(options.Quality, 0, 100), threads);
        (byte[] y, byte[] u, byte[] v) = RgbToI420.AllocatePlanes(clip.Width, clip.Height);
        var packets = new List<VideoPacket>();
        for (int i = 0; i < clip.Frames.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            RgbToI420.Convert(clip.Frames[i], y, u, v);
            encoder.Encode(y, u, v, clip.Width, i);
            encoder.DrainPackets(packets);
        }

        encoder.Flush();
        encoder.DrainPackets(packets);
        return WebMMuxer.Write(_codec, clip.Width, clip.Height, numerator, denominator, clip.Frames.Count, packets);
    }
}
