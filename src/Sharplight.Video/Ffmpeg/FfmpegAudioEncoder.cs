using FFmpeg.AutoGen;

namespace Sharplight.Video.Ffmpeg;

/// <summary>
/// The encoded sound track of an encode: an audio encoder and its stream, fed through a sample FIFO in the encoder's format (filled by
/// the caller) and sent frame by frame up to a sample position, so the track is interleaved with the video as it is written.
/// </summary>
internal sealed unsafe class FfmpegAudioEncoder : IDisposable
{
    private readonly FfmpegMuxer _muxer;
    private AVCodecContext* _encoder;
    private AVStream* _stream;
    private AVAudioFifo* _fifo;
    private AVFrame* _frame;

    /// <summary>Opens the encoder and adds its stream (before the muxer's header is written).</summary>
    /// <param name="muxer">The output.</param>
    /// <param name="plan">The codec and bitrate.</param>
    /// <param name="sourceSampleRate">The rate of the source (the encoder takes it, or the nearest it supports).</param>
    /// <param name="sourceChannels">The source's channels (the track has at most two).</param>
    public FfmpegAudioEncoder(FfmpegMuxer muxer, FfmpegAudioPlan plan, int sourceSampleRate, int sourceChannels)
    {
        _muxer = muxer;
        AVCodec* codec = ffmpeg.avcodec_find_encoder_by_name(plan.EncoderName);
        if (codec is null)
        {
            throw new PlatformNotSupportedException($"The bundled FFmpeg has no {plan.EncoderName} encoder.");
        }

        _encoder = ffmpeg.avcodec_alloc_context3(codec);
        int channels = Math.Clamp(sourceChannels, 1, 2);
        ffmpeg.av_channel_layout_default(&_encoder->ch_layout, channels);
        _encoder->sample_rate = PickSampleRate(codec, sourceSampleRate);
        _encoder->sample_fmt = PickSampleFormat(codec);
        _encoder->time_base = new AVRational { num = 1, den = _encoder->sample_rate };
        if (plan.BitRate > 0)
        {
            _encoder->bit_rate = plan.BitRate;
        }

        if (muxer.NeedsGlobalHeader)
        {
            _encoder->flags |= ffmpeg.AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        muxer.Check(ffmpeg.avcodec_open2(_encoder, codec, null), $"open the {plan.EncoderName} encoder");
        _stream = ffmpeg.avformat_new_stream(muxer.Format, null);
        muxer.Check(ffmpeg.avcodec_parameters_from_context(_stream->codecpar, _encoder), "describe the audio stream");
        _stream->time_base = _encoder->time_base;
        _fifo = ffmpeg.av_audio_fifo_alloc(_encoder->sample_fmt, channels, Math.Max(1, FrameSize));
        _frame = ffmpeg.av_frame_alloc();
        _frame->format = (int)_encoder->sample_fmt;
        _frame->sample_rate = _encoder->sample_rate;
        _frame->nb_samples = FrameSize;
        muxer.Check(ffmpeg.av_channel_layout_copy(&_frame->ch_layout, &_encoder->ch_layout), "set the channel layout");
        muxer.Check(ffmpeg.av_frame_get_buffer(_frame, 0), "allocate an audio frame");
    }

    /// <summary>The encoder's sample rate in Hz.</summary>
    public int SampleRate => _encoder->sample_rate;

    /// <summary>The encoder's sample format.</summary>
    public AVSampleFormat SampleFormat => _encoder->sample_fmt;

    /// <summary>The encoder's channel layout.</summary>
    public AVChannelLayout* Layout => &_encoder->ch_layout;

    /// <summary>The samples of one encoded frame.</summary>
    public int FrameSize => _encoder->frame_size > 0 ? _encoder->frame_size : 1024;

    /// <summary>The samples sent to the encoder so far.</summary>
    public long Sent { get; private set; }

    /// <summary>The samples sent plus those waiting in the FIFO.</summary>
    public long Available => Sent + ffmpeg.av_audio_fifo_size(_fifo);

    /// <summary>Appends samples in the encoder's format to the FIFO, at most up to <paramref name="limit"/> available samples.</summary>
    /// <param name="samples">The planes.</param>
    /// <param name="count">Samples per channel.</param>
    /// <param name="limit">The track's length cap in samples.</param>
    public void Append(byte** samples, int count, long limit = long.MaxValue)
    {
        int take = (int)Math.Clamp(limit - Available, 0, count);
        if (take > 0)
        {
            _muxer.Check(ffmpeg.av_audio_fifo_write(_fifo, (void**)samples, take), "buffer the audio");
        }
    }

    /// <summary>
    /// Sends whole frames while fewer than <paramref name="until"/> samples are sent. A frame needing samples past <paramref name="total"/>
    /// (the track's length) waits, unless <paramref name="final"/>: then the last frame is sent short (or padded with silence for encoders
    /// with a fixed frame size). Missing FIFO samples are silence.
    /// </summary>
    public void SendUntil(long until, long total, bool final)
    {
        int frameSize = FrameSize;
        bool variable = (_encoder->codec->capabilities & (ffmpeg.AV_CODEC_CAP_VARIABLE_FRAME_SIZE | ffmpeg.AV_CODEC_CAP_SMALL_LAST_FRAME)) != 0;
        while (Sent < Math.Min(until, total))
        {
            long remaining = total - Sent;
            if (remaining < frameSize && !final)
            {
                return;
            }

            _muxer.Check(ffmpeg.av_frame_make_writable(_frame), "reuse the audio frame");
            int count = (int)Math.Min(frameSize, remaining);
            void** data = (void**)&_frame->data;
            int read = ffmpeg.av_audio_fifo_read(_fifo, data, count);
            int samples = Math.Max(read, 0);
            if (samples < count)
            {
                ffmpeg.av_samples_set_silence((byte**)data, samples, count - samples, _encoder->ch_layout.nb_channels, _encoder->sample_fmt);
            }

            int sent = variable ? count : frameSize;
            if (sent > count)
            {
                ffmpeg.av_samples_set_silence((byte**)data, count, sent - count, _encoder->ch_layout.nb_channels, _encoder->sample_fmt);
            }

            _frame->nb_samples = sent;
            _frame->pts = Sent;
            Sent += count;
            _muxer.Send(_encoder, _stream, _frame);
        }
    }

    /// <summary>Drains the encoder.</summary>
    public void Flush() => _muxer.Send(_encoder, _stream, null);

    /// <inheritdoc />
    public void Dispose()
    {
        if (_fifo is not null)
        {
            ffmpeg.av_audio_fifo_free(_fifo);
            _fifo = null;
        }

        AVFrame* frame = _frame;
        ffmpeg.av_frame_free(&frame);
        _frame = null;
        AVCodecContext* encoder = _encoder;
        ffmpeg.avcodec_free_context(&encoder);
        _encoder = null;
    }

    private static int PickSampleRate(AVCodec* codec, int preferred)
    {
        int* rates = null;
        int count = 0;
        ffmpeg.avcodec_get_supported_config(null, codec, AVCodecConfig.AV_CODEC_CONFIG_SAMPLE_RATE, 0, (void**)&rates, &count);
        if (rates is null || count == 0)
        {
            return preferred;
        }

        var supported = new ReadOnlySpan<int>(rates, count).ToArray();
        if (supported.Contains(preferred))
        {
            return preferred;
        }

        // The lowest supported rate at or above the source, else the highest.
        int[] higher = [.. supported.Where(r => r >= preferred).Order()];
        return higher.Length > 0 ? higher[0] : supported.Max();
    }

    private static AVSampleFormat PickSampleFormat(AVCodec* codec)
    {
        AVSampleFormat* formats = null;
        int count = 0;
        ffmpeg.avcodec_get_supported_config(null, codec, AVCodecConfig.AV_CODEC_CONFIG_SAMPLE_FORMAT, 0, (void**)&formats, &count);
        if (formats is null || count == 0)
        {
            return AVSampleFormat.AV_SAMPLE_FMT_FLTP;
        }

        var supported = new ReadOnlySpan<AVSampleFormat>(formats, count);
        return supported.Contains(AVSampleFormat.AV_SAMPLE_FMT_FLTP) ? AVSampleFormat.AV_SAMPLE_FMT_FLTP : supported[0];
    }
}
