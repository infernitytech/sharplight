using Microsoft.Extensions.Options;
using Sharplight.Video.Ffmpeg;

namespace Sharplight.Video;

/// <summary>
/// Encodes one format (<see cref="VideoFormats.Mp4"/>, <see cref="VideoFormats.WebM"/>, <see cref="VideoFormats.Mov"/>,
/// <see cref="VideoFormats.Matroska"/>, <see cref="VideoFormats.Gif"/> or <see cref="VideoFormats.Apng"/>) with the FFmpeg libraries
/// Sharplight ships (<see cref="BundledFfmpeg"/>): H.264 (OpenH264), AV1 (SVT-AV1), VP9 / VP8 (libvpx), ProRes, FFV1, APNG, GIF with an
/// optimized palette, the hardware encoders of the machine, and an AAC / Opus / FLAC / PCM sound track from <see cref="VideoClip.Audio"/>.
/// Takes precedence over the managed GIF and APNG encoders (<see cref="Priority"/>); <see cref="IsAvailable"/> is false when the libraries
/// are missing, and the catalog then falls back to the managed encoders.
/// </summary>
public sealed class FfmpegVideoEncoder : IVideoEncoder
{
    /// <summary>The <see cref="Priority"/> of the bundled encoders (the managed ones have 0).</summary>
    public const int BundledPriority = 10;

    private readonly FfmpegContainer _container;
    private readonly bool _hardware;

    /// <summary>Creates the encoder of one format.</summary>
    /// <param name="format">The format id (<see cref="VideoFormats"/>).</param>
    /// <param name="options">Whether hardware encoders are offered; <see langword="null"/> uses the defaults.</param>
    /// <exception cref="ArgumentException">The format is not one of the bundled formats.</exception>
    public FfmpegVideoEncoder(string format, IOptions<SharplightVideoOptions>? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(format);
        _container = FfmpegContainers.ForFormat(format);
        _hardware = options?.Value.EnableHardwareEncoders ?? true;
    }

    /// <summary>The formats this encoder class handles (one instance per format).</summary>
    public static IReadOnlyList<string> Formats { get; } = [.. FfmpegContainers.All.Select(c => c.Format)];

    /// <inheritdoc />
    public string Format => _container.Format;

    /// <inheritdoc />
    public IReadOnlyList<string> FileExtensions => _container.FileExtensions;

    /// <inheritdoc />
    public bool IsAvailable => BundledFfmpeg.IsAvailable;

    /// <inheritdoc />
    public int Priority => BundledPriority;

    /// <inheritdoc />
    /// <remarks>The first call tries the hardware encoders once (tens of milliseconds); call it off the UI thread.</remarks>
    public IReadOnlyList<VideoCodecInfo> Codecs => IsAvailable ? [.. UsableCodecs().Select(c => c.ToInfo(_container.AudioCodecs))] : [];

    /// <inheritdoc />
    public async Task EncodeAsync(VideoClip clip, Stream output, VideoEncodeOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(clip);
        ArgumentNullException.ThrowIfNull(output);
        BundledFfmpeg.ThrowIfUnavailable();
        VideoEncodeOptions settings = options ?? new VideoEncodeOptions();
        await Task.Run(
            () =>
            {
                FfmpegEncodePlan plan = FfmpegEncodePlanner.Create(_container, UsableCodecs(), clip.Width, clip.Height, clip.FramesPerSecond, clip.Audio is not null, settings,
                    output.CanSeek);
                FfmpegVideoWriter.Write(plan, clip, output, cancellationToken);
            },
            cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc />
    /// <remarks>True for every format but GIF, whose palette is computed from all frames (its writer would hold the whole clip).</remarks>
    public bool SupportsIncrementalWriting => _container.Format != VideoFormats.Gif;

    /// <inheritdoc />
    /// <remarks>The first call tries the hardware encoders once (tens of milliseconds); call it off the UI thread.</remarks>
    public IVideoFrameWriter OpenWriter(Stream output, VideoStreamSettings stream, VideoEncodeOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(stream);
        stream.Validate();
        if (!SupportsIncrementalWriting)
        {
            throw new NotSupportedException($"The {Format} encoder writes whole clips only.");
        }

        BundledFfmpeg.ThrowIfUnavailable();
        VideoEncodeOptions settings = options ?? new VideoEncodeOptions();
        FfmpegEncodePlan plan = FfmpegEncodePlanner.Create(_container, UsableCodecs(), stream.Width, stream.Height, stream.FramesPerSecond, stream.AudioSourcePath is not null,
            settings, output.CanSeek);
        return new FfmpegVideoFrameWriter(FfmpegVideoWriter.Open(plan, output, stream.AudioSourcePath, stream.AudioSourceStart), stream.Width, stream.Height);
    }

    private IReadOnlyList<FfmpegCodec> UsableCodecs() => FfmpegCodecProbe.Usable(_container, _hardware);
}
