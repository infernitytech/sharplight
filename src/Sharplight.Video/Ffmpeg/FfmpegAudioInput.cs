using FFmpeg.AutoGen;

namespace Sharplight.Video.Ffmpeg;

/// <summary>
/// The first audio stream of a media file read packet by packet from a start time on (the sound track source of an incremental encode).
/// Timestamps are measured from the start: <see cref="Seconds"/> of a packet is its time after the start (negative before it).
/// </summary>
internal sealed unsafe class FfmpegAudioInput : IDisposable
{
    private readonly long _origin;
    private AVFormatContext* _format;

    private FfmpegAudioInput(AVFormatContext* format, int streamIndex, TimeSpan start)
    {
        _format = format;
        StreamIndex = streamIndex;
        AVStream* stream = Stream;
        long streamStart = stream->start_time != ffmpeg.AV_NOPTS_VALUE ? stream->start_time : 0;
        _origin = streamStart + (long)Math.Round(start.TotalSeconds / ffmpeg.av_q2d(stream->time_base));
        if (start > TimeSpan.Zero && ffmpeg.av_seek_frame(_format, streamIndex, _origin, ffmpeg.AVSEEK_FLAG_BACKWARD) < 0)
        {
            // Not seekable: read from the beginning; the packets before the start are skipped by their timestamps.
            ffmpeg.av_seek_frame(_format, streamIndex, streamStart, ffmpeg.AVSEEK_FLAG_BACKWARD);
        }
    }

    /// <summary>The audio stream.</summary>
    public AVStream* Stream => _format->streams[StreamIndex];

    /// <summary>The stream's index in the file.</summary>
    public int StreamIndex { get; }

    /// <summary>Opens the file's best audio stream.</summary>
    /// <param name="path">The media file.</param>
    /// <param name="start">Where the track starts.</param>
    /// <returns>The input, or <see langword="null"/> when the file has no audio stream.</returns>
    /// <exception cref="InvalidDataException">The file cannot be read.</exception>
    public static FfmpegAudioInput? Open(string path, TimeSpan start)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("The audio source does not exist.", path);
        }

        AVFormatContext* format = null;
        FfmpegError.CheckInput(ffmpeg.avformat_open_input(&format, path, null, null), $"open '{Path.GetFileName(path)}'");
        try
        {
            FfmpegError.CheckInput(ffmpeg.avformat_find_stream_info(format, null), $"read the streams of '{Path.GetFileName(path)}'");
            int index = ffmpeg.av_find_best_stream(format, AVMediaType.AVMEDIA_TYPE_AUDIO, -1, -1, null, 0);
            if (index >= 0)
            {
                return new FfmpegAudioInput(format, index, start);
            }
        }
        catch
        {
            ffmpeg.avformat_close_input(&format);
            throw;
        }

        ffmpeg.avformat_close_input(&format);
        return null;
    }

    /// <summary>Reads the next packet of the audio stream.</summary>
    /// <param name="packet">Receives the packet (the caller unreferences it).</param>
    /// <returns>False at the end of the file.</returns>
    public bool Read(AVPacket* packet)
    {
        while (ffmpeg.av_read_frame(_format, packet) >= 0)
        {
            if (packet->stream_index == StreamIndex)
            {
                return true;
            }

            ffmpeg.av_packet_unref(packet);
        }

        return false;
    }

    /// <summary>A timestamp of the stream in ticks after the start (negative before it).</summary>
    public long Ticks(long timestamp) => timestamp - _origin;

    /// <summary>A timestamp of the stream in seconds after the start (negative before it).</summary>
    public double Seconds(long timestamp) => Ticks(timestamp) * ffmpeg.av_q2d(Stream->time_base);

    /// <inheritdoc />
    public void Dispose()
    {
        AVFormatContext* format = _format;
        ffmpeg.avformat_close_input(&format);
        _format = null;
    }
}
