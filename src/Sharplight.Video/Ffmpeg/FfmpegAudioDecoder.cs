using FFmpeg.AutoGen;

namespace Sharplight.Video.Ffmpeg;

/// <summary>
/// Decodes a source audio stream (<see cref="FfmpegAudioInput"/>) and resamples it into an <see cref="FfmpegAudioEncoder"/>'s FIFO a packet
/// at a time, dropping the samples before the start: the transcoded sound track of an incremental encode, with memory bounded by what the
/// caller asks for.
/// </summary>
internal sealed unsafe class FfmpegAudioDecoder : IDisposable
{
    private readonly FfmpegAudioInput _input;
    private readonly FfmpegAudioEncoder _target;
    private AVCodecContext* _decoder;
    private SwrContext* _resampler;
    private AVPacket* _packet;
    private AVFrame* _frame;
    private bool _draining;
    private bool _started;

    /// <param name="input">The source stream (owned by the caller).</param>
    /// <param name="target">The encoder whose FIFO receives the samples.</param>
    /// <exception cref="InvalidDataException">The bundled FFmpeg cannot decode the source.</exception>
    public FfmpegAudioDecoder(FfmpegAudioInput input, FfmpegAudioEncoder target)
    {
        _input = input;
        _target = target;
        AVCodecParameters* parameters = input.Stream->codecpar;
        AVCodec* codec = ffmpeg.avcodec_find_decoder(parameters->codec_id);
        if (codec is null)
        {
            throw new InvalidDataException("The bundled FFmpeg has no decoder for the audio source.");
        }

        _decoder = ffmpeg.avcodec_alloc_context3(codec);
        _packet = ffmpeg.av_packet_alloc();
        _frame = ffmpeg.av_frame_alloc();
        FfmpegError.CheckInput(ffmpeg.avcodec_parameters_to_context(_decoder, parameters), "read the audio parameters");
        _decoder->pkt_timebase = input.Stream->time_base;
        FfmpegError.CheckInput(ffmpeg.avcodec_open2(_decoder, codec, null), "open the audio decoder");
    }

    /// <summary>Whether the source has no more samples.</summary>
    public bool Ended { get; private set; }

    /// <summary>Decodes until the target holds at least <paramref name="samples"/> samples (sent and buffered) or the source ends.</summary>
    public void FillTo(long samples)
    {
        while (!Ended && _target.Available < samples)
        {
            Step();
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        SwrContext* resampler = _resampler;
        ffmpeg.swr_free(&resampler);
        _resampler = null;
        AVFrame* frame = _frame;
        ffmpeg.av_frame_free(&frame);
        _frame = null;
        AVPacket* packet = _packet;
        ffmpeg.av_packet_free(&packet);
        _packet = null;
        AVCodecContext* decoder = _decoder;
        ffmpeg.avcodec_free_context(&decoder);
        _decoder = null;
    }

    /// <summary>Feeds one packet (or the end of the stream) and converts the frames that come out.</summary>
    private void Step()
    {
        if (!_draining)
        {
            if (_input.Read(_packet))
            {
                int sent = ffmpeg.avcodec_send_packet(_decoder, _packet);
                ffmpeg.av_packet_unref(_packet);
                if (sent < 0 && sent != ffmpeg.AVERROR(ffmpeg.EAGAIN) && sent != ffmpeg.AVERROR_INVALIDDATA)
                {
                    FfmpegError.CheckInput(sent, "decode the audio");
                }
            }
            else
            {
                _draining = true;
                ffmpeg.avcodec_send_packet(_decoder, null);
            }
        }

        int received;
        while ((received = ffmpeg.avcodec_receive_frame(_decoder, _frame)) >= 0)
        {
            Convert(_frame);
            ffmpeg.av_frame_unref(_frame);
        }

        if (_draining && received != ffmpeg.AVERROR(ffmpeg.EAGAIN))
        {
            if (_resampler is not null)
            {
                Resample(null, 0, 0);
            }

            Ended = true;
        }
    }

    private void Convert(AVFrame* frame)
    {
        long timestamp = frame->best_effort_timestamp != ffmpeg.AV_NOPTS_VALUE ? frame->best_effort_timestamp : frame->pts;
        double time = timestamp == ffmpeg.AV_NOPTS_VALUE ? 0 : _input.Seconds(timestamp);
        double duration = frame->nb_samples / (double)Math.Max(1, frame->sample_rate);
        if (!_started && time + duration <= 0)
        {
            return;
        }

        int skip = 0;
        if (!_started)
        {
            // The first frame that reaches the start: drop its samples before it (in output samples).
            skip = time < 0 ? (int)Math.Round(-time * _target.SampleRate) : 0;
            _started = true;
        }

        if (_resampler is null)
        {
            SwrContext* resampler = null;
            FfmpegError.Check(ffmpeg.swr_alloc_set_opts2(&resampler, _target.Layout, _target.SampleFormat, _target.SampleRate, &frame->ch_layout, (AVSampleFormat)frame->format,
                frame->sample_rate, 0, null), "set up the resampler");
            _resampler = resampler;
            FfmpegError.Check(ffmpeg.swr_init(_resampler), "set up the resampler");
        }

        Resample(frame->extended_data, frame->nb_samples, skip);
    }

    private void Resample(byte** input, int count, int skip)
    {
        int capacity = ffmpeg.swr_get_out_samples(_resampler, count) + 32;
        int channels = _target.Layout->nb_channels;
        byte** output = stackalloc byte*[8];
        int lineSize;
        FfmpegError.Check(ffmpeg.av_samples_alloc(output, &lineSize, channels, capacity, _target.SampleFormat, 0), "allocate audio samples");
        try
        {
            int converted = FfmpegError.Check(ffmpeg.swr_convert(_resampler, output, capacity, input, count), "resample the audio");
            int drop = Math.Min(skip, converted);
            if (converted > drop)
            {
                byte** kept = stackalloc byte*[8];
                bool planar = ffmpeg.av_sample_fmt_is_planar(_target.SampleFormat) != 0;
                int bytesPerSample = ffmpeg.av_get_bytes_per_sample(_target.SampleFormat) * (planar ? 1 : channels);
                for (int plane = 0; plane < (planar ? channels : 1); plane++)
                {
                    kept[plane] = output[plane] + drop * bytesPerSample;
                }

                _target.Append(kept, converted - drop);
            }
        }
        finally
        {
            ffmpeg.av_freep(&output[0]);
        }
    }
}
