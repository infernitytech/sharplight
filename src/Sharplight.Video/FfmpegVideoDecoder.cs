using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Sharplight.Audio;
using Sharplight.Video.Ffmpeg;

namespace Sharplight.Video;

/// <summary>
/// Reads video files with the FFmpeg libraries Sharplight ships (<see cref="BundledFfmpeg"/>): H.264, HEVC, VP8, VP9, AV1 (dav1d), ProRes,
/// MPEG-1/2/4, FFV1, GIF, APNG and more in MP4 / MOV, Matroska / WebM, AVI, MPEG-TS, Ogg and image sequences, with AAC, MP3, Opus, Vorbis,
/// FLAC, AC-3 and PCM audio. Decoding runs on the thread pool.
/// </summary>
public sealed class FfmpegVideoDecoder : IVideoDecoder
{
    private const int FrameQueueLength = 4;

    /// <inheritdoc />
    public bool IsAvailable => BundledFfmpeg.IsAvailable;

    /// <inheritdoc />
    public IReadOnlyList<string> FileExtensions { get; } =
        [".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".gif", ".apng", ".mpg", ".mpeg", ".ts", ".m2ts", ".flv", ".ogv", ".ivf", ".y4m"];

    /// <inheritdoc />
    public Task<VideoInfo> ProbeAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        BundledFfmpeg.ThrowIfUnavailable();
        return Task.Run(
            () =>
            {
                using var reader = new FfmpegVideoReader(path);
                return reader.Probe();
            },
            cancellationToken);
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<VideoFrame> ReadFramesAsync(string path, VideoDecodeOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        BundledFfmpeg.ThrowIfUnavailable();
        VideoDecodeOptions settings = options ?? new VideoDecodeOptions();
        var channel = Channel.CreateBounded<VideoFrame>(new BoundedChannelOptions(FrameQueueLength) { SingleReader = true, SingleWriter = true });
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        Task producer = Task.Run(() => Produce(path, settings, channel.Writer, stop.Token), CancellationToken.None);
        try
        {
            // A decode error completes the channel with the exception, so it surfaces here.
            await foreach (VideoFrame frame in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                yield return frame;
            }
        }
        finally
        {
            // Stops the decoder when the consumer leaves early and waits for it to release the native resources.
            await stop.CancelAsync().ConfigureAwait(false);
            await producer.ConfigureAwait(ConfigureAwaitOptions.SuppressThrowing);
        }
    }

    /// <inheritdoc />
    public Task<AudioBuffer?> ReadAudioAsync(string path, int? sampleRate = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (sampleRate is int rate)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rate, nameof(sampleRate));
        }

        BundledFfmpeg.ThrowIfUnavailable();
        return Task.Run(
            () =>
            {
                using var reader = new FfmpegVideoReader(path);
                return reader.ReadAudio(sampleRate, cancellationToken);
            },
            cancellationToken);
    }

    private static void Produce(string path, VideoDecodeOptions options, ChannelWriter<VideoFrame> writer, CancellationToken cancellationToken)
    {
        try
        {
            using var reader = new FfmpegVideoReader(path);
            reader.ReadFrames(options, frame => Write(writer, frame, cancellationToken), cancellationToken);
            writer.Complete();
        }
        catch (Exception exception)
        {
            writer.Complete(exception);
        }
    }

    private static void Write(ChannelWriter<VideoFrame> writer, VideoFrame frame, CancellationToken cancellationToken)
    {
        while (!writer.TryWrite(frame))
        {
            if (!writer.WaitToWriteAsync(cancellationToken).AsTask().GetAwaiter().GetResult())
            {
                throw new OperationCanceledException(cancellationToken);
            }
        }
    }
}
