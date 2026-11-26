using FFmpeg.AutoGen;

namespace Sharplight.Video.Ffmpeg;

/// <summary>
/// A sound track copied from a source file packet by packet (no decoding): the packets from the start time on, shifted to begin at zero
/// and written interleaved with the video up to a time, so the track ends with the video.
/// </summary>
internal sealed unsafe class FfmpegAudioCopy : IDisposable
{
    private readonly FfmpegMuxer _muxer;
    private readonly FfmpegAudioInput _input;
    private readonly bool _skipBeforeStart;
    private AVStream* _stream;
    private AVPacket* _pending;
    private bool _hasPending;
    private bool _ended;

    /// <summary>Adds the copied stream to the muxer (before its header is written).</summary>
    /// <param name="muxer">The output; it must accept the source's codec (<see cref="FfmpegMuxer.Accepts"/>).</param>
    /// <param name="input">The source stream (owned by the caller).</param>
    /// <param name="skipBeforeStart">Whether packets before the start are dropped (a start inside the file).</param>
    public FfmpegAudioCopy(FfmpegMuxer muxer, FfmpegAudioInput input, bool skipBeforeStart)
    {
        _muxer = muxer;
        _input = input;
        _skipBeforeStart = skipBeforeStart;
        _stream = ffmpeg.avformat_new_stream(muxer.Format, null);
        muxer.Check(ffmpeg.avcodec_parameters_copy(_stream->codecpar, input.Stream->codecpar), "copy the audio parameters");
        _stream->codecpar->codec_tag = 0;
        _stream->time_base = input.Stream->time_base;
        _pending = ffmpeg.av_packet_alloc();
    }

    /// <summary>Writes the packets that start before <paramref name="seconds"/> after the track's start.</summary>
    public void WriteUntil(double seconds)
    {
        AVRational sourceBase = _input.Stream->time_base;
        while (!_ended)
        {
            if (!_hasPending)
            {
                _hasPending = _input.Read(_pending);
                if (!_hasPending)
                {
                    _ended = true;
                    return;
                }
            }

            long timestamp = _pending->pts != ffmpeg.AV_NOPTS_VALUE ? _pending->pts : _pending->dts;
            double time = timestamp == ffmpeg.AV_NOPTS_VALUE ? 0 : _input.Seconds(timestamp);
            if (time >= seconds)
            {
                return;
            }

            _hasPending = false;
            if (_skipBeforeStart && time < 0)
            {
                ffmpeg.av_packet_unref(_pending);
                continue;
            }

            if (_pending->pts != ffmpeg.AV_NOPTS_VALUE)
            {
                _pending->pts = _input.Ticks(_pending->pts);
            }

            if (_pending->dts != ffmpeg.AV_NOPTS_VALUE)
            {
                _pending->dts = _input.Ticks(_pending->dts);
            }

            ffmpeg.av_packet_rescale_ts(_pending, sourceBase, _stream->time_base);
            _pending->stream_index = _stream->index;
            _pending->pos = -1;
            _muxer.Write(_pending);
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        AVPacket* pending = _pending;
        ffmpeg.av_packet_free(&pending);
        _pending = null;
        _stream = null;
    }
}
