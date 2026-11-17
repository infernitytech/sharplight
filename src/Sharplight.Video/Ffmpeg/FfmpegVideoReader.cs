using FFmpeg.AutoGen;
using Sharplight.Audio;

namespace Sharplight.Video.Ffmpeg;

/// <summary>
/// Reads a video file with the bundled FFmpeg: stream properties, the frames of the first video stream converted by swscale (honoring the
/// stream's YUV matrix and range), and the first audio stream resampled to float. One instance per read; not thread safe.
/// </summary>
internal sealed unsafe class FfmpegVideoReader : IDisposable
{
    private AVFormatContext* _format;
    private AVCodecContext* _decoder;
    private AVPacket* _packet;
    private AVFrame* _frame;
    private SwsContext* _scale;
    private (int Width, int Height, AVPixelFormat Format, AVColorSpace Space, AVColorRange Range) _scaleSource;

    /// <param name="path">The file.</param>
    /// <exception cref="InvalidDataException">The file cannot be opened or has no streams.</exception>
    public FfmpegVideoReader(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("The video file does not exist.", path);
        }

        AVFormatContext* format = null;
        FfmpegError.CheckInput(ffmpeg.avformat_open_input(&format, path, null, null), $"open '{Path.GetFileName(path)}'");
        _format = format;
        FfmpegError.CheckInput(ffmpeg.avformat_find_stream_info(_format, null), $"read the streams of '{Path.GetFileName(path)}'");
        _packet = ffmpeg.av_packet_alloc();
        _frame = ffmpeg.av_frame_alloc();
    }

    /// <summary>The properties of the first video and audio streams.</summary>
    /// <exception cref="InvalidDataException">The file has no video stream.</exception>
    public VideoInfo Probe()
    {
        int videoIndex = ffmpeg.av_find_best_stream(_format, AVMediaType.AVMEDIA_TYPE_VIDEO, -1, -1, null, 0);
        if (videoIndex < 0)
        {
            throw new InvalidDataException("The file has no video stream.");
        }

        AVStream* video = _format->streams[videoIndex];
        double rate = FrameRateOf(video);
        double seconds = video->duration > 0 ? video->duration * ffmpeg.av_q2d(video->time_base) : _format->duration > 0 ? _format->duration / (double)ffmpeg.AV_TIME_BASE : 0;
        long frames = video->nb_frames > 0 ? video->nb_frames : (long)Math.Round(seconds * rate);
        var info = new VideoInfo
        {
            Container = ToString(_format->iformat->name),
            VideoCodec = ffmpeg.avcodec_get_name(video->codecpar->codec_id),
            Width = video->codecpar->width,
            Height = video->codecpar->height,
            FramesPerSecond = rate,
            Duration = TimeSpan.FromSeconds(seconds),
            FrameCount = frames,
        };

        int audioIndex = ffmpeg.av_find_best_stream(_format, AVMediaType.AVMEDIA_TYPE_AUDIO, -1, videoIndex, null, 0);
        if (audioIndex < 0)
        {
            return info;
        }

        AVCodecParameters* audio = _format->streams[audioIndex]->codecpar;
        return info with { AudioCodec = ffmpeg.avcodec_get_name(audio->codec_id), AudioSampleRate = audio->sample_rate, AudioChannels = audio->ch_layout.nb_channels };
    }

    /// <summary>Decodes the frames of the first video stream and hands them to <paramref name="emit"/> in presentation order.</summary>
    /// <param name="options">Range, size, rate and layout.</param>
    /// <param name="emit">Receives each frame (on the calling thread).</param>
    /// <param name="cancellationToken">Checked between packets.</param>
    public void ReadFrames(VideoDecodeOptions options, Action<VideoFrame> emit, CancellationToken cancellationToken)
    {
        AVStream* stream = OpenDecoder(AVMediaType.AVMEDIA_TYPE_VIDEO);
        int streamIndex = stream->index;
        double timeBase = ffmpeg.av_q2d(stream->time_base);
        double sourceRate = FrameRateOf(stream);
        double start = options.Start.TotalSeconds;
        double end = options.Duration is TimeSpan duration ? start + duration.TotalSeconds : double.PositiveInfinity;
        double startOffset = stream->start_time != ffmpeg.AV_NOPTS_VALUE ? stream->start_time * timeBase : 0;
        if (start > 0 && ffmpeg.av_seek_frame(_format, streamIndex, (long)((start + startOffset) / timeBase), ffmpeg.AVSEEK_FLAG_BACKWARD) >= 0)
        {
            ffmpeg.avcodec_flush_buffers(_decoder);
        }

        var sequencer = new FrameSequencer(options.FramesPerSecond, start, options.MaxFrames ?? int.MaxValue, emit);
        double lastTime = start;
        bool draining = false;
        while (!sequencer.IsFull)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!draining)
            {
                int read = ffmpeg.av_read_frame(_format, _packet);
                if (read < 0)
                {
                    draining = true;
                    ffmpeg.avcodec_send_packet(_decoder, null);
                }
                else
                {
                    bool ours = _packet->stream_index == streamIndex;
                    int sent = ours ? ffmpeg.avcodec_send_packet(_decoder, _packet) : 0;
                    ffmpeg.av_packet_unref(_packet);
                    if (!ours)
                    {
                        continue;
                    }

                    if (sent < 0 && sent != ffmpeg.AVERROR(ffmpeg.EAGAIN))
                    {
                        FfmpegError.CheckInput(sent, "decode the video");
                    }
                }
            }

            bool ended = false;
            while (!sequencer.IsFull)
            {
                int received = ffmpeg.avcodec_receive_frame(_decoder, _frame);
                if (received == ffmpeg.AVERROR(ffmpeg.EAGAIN))
                {
                    break;
                }

                if (received == ffmpeg.AVERROR_EOF)
                {
                    ended = true;
                    break;
                }

                FfmpegError.CheckInput(received, "decode the video");
                long pts = _frame->best_effort_timestamp != ffmpeg.AV_NOPTS_VALUE ? _frame->best_effort_timestamp : _frame->pts;
                double time = pts == ffmpeg.AV_NOPTS_VALUE ? lastTime + 1 / Math.Max(sourceRate, 1) : pts * timeBase - startOffset;
                bool beforeStart = time + 1e-6 < start;
                if (time >= end)
                {
                    ended = true;
                }
                else if (!beforeStart)
                {
                    lastTime = time;
                    sequencer.Add(Convert(_frame, options), time);
                }

                ffmpeg.av_frame_unref(_frame);
                if (ended)
                {
                    break;
                }
            }

            if (ended)
            {
                break;
            }
        }

        double lastFrameEnd = lastTime + (sourceRate > 0 ? 1 / sourceRate : 0);
        sequencer.Finish(Math.Min(end, lastFrameEnd));
    }

    /// <summary>Decodes the first audio stream to interleaved float samples.</summary>
    /// <param name="sampleRate">The output rate, or <see langword="null"/> for the source rate.</param>
    /// <param name="cancellationToken">Checked between packets.</param>
    /// <returns>The audio, or <see langword="null"/> without an audio stream.</returns>
    public AudioBuffer? ReadAudio(int? sampleRate, CancellationToken cancellationToken)
    {
        if (ffmpeg.av_find_best_stream(_format, AVMediaType.AVMEDIA_TYPE_AUDIO, -1, -1, null, 0) < 0)
        {
            return null;
        }

        AVStream* stream = OpenDecoder(AVMediaType.AVMEDIA_TYPE_AUDIO);
        int channels = Math.Max(1, _decoder->ch_layout.nb_channels);
        int rate = sampleRate ?? _decoder->sample_rate;
        AVChannelLayout layout = default;
        ffmpeg.av_channel_layout_default(&layout, channels);
        SwrContext* resampler = null;
        var samples = new List<float>();
        try
        {
            FfmpegError.Check(ffmpeg.swr_alloc_set_opts2(&resampler, &layout, AVSampleFormat.AV_SAMPLE_FMT_FLT, rate, &_decoder->ch_layout, _decoder->sample_fmt, _decoder->sample_rate, 0, null), "set up the resampler");
            FfmpegError.Check(ffmpeg.swr_init(resampler), "set up the resampler");
            bool draining = false;
            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!draining)
                {
                    if (ffmpeg.av_read_frame(_format, _packet) < 0)
                    {
                        draining = true;
                        ffmpeg.avcodec_send_packet(_decoder, null);
                    }
                    else
                    {
                        if (_packet->stream_index == stream->index)
                        {
                            ffmpeg.avcodec_send_packet(_decoder, _packet);
                        }

                        ffmpeg.av_packet_unref(_packet);
                    }
                }

                int received;
                while ((received = ffmpeg.avcodec_receive_frame(_decoder, _frame)) >= 0)
                {
                    AppendConverted(resampler, _frame->extended_data, _frame->nb_samples, channels, samples);
                    ffmpeg.av_frame_unref(_frame);
                }

                if (received == ffmpeg.AVERROR_EOF || (draining && received != ffmpeg.AVERROR(ffmpeg.EAGAIN)))
                {
                    break;
                }
            }

            AppendConverted(resampler, null, 0, channels, samples);
        }
        finally
        {
            ffmpeg.swr_free(&resampler);
            ffmpeg.av_channel_layout_uninit(&layout);
        }

        return samples.Count == 0 ? null : new AudioBuffer([.. samples], rate, channels);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_scale is not null)
        {
            ffmpeg.sws_freeContext(_scale);
            _scale = null;
        }

        AVFrame* frame = _frame;
        ffmpeg.av_frame_free(&frame);
        AVPacket* packet = _packet;
        ffmpeg.av_packet_free(&packet);
        AVCodecContext* decoder = _decoder;
        ffmpeg.avcodec_free_context(&decoder);
        AVFormatContext* format = _format;
        ffmpeg.avformat_close_input(&format);
        _frame = null;
        _packet = null;
        _decoder = null;
        _format = null;
    }

    private AVStream* OpenDecoder(AVMediaType type)
    {
        AVCodec* codec = null;
        int index = ffmpeg.av_find_best_stream(_format, type, -1, -1, &codec, 0);
        string kind = type == AVMediaType.AVMEDIA_TYPE_VIDEO ? "video" : "audio";
        if (index < 0 || codec is null)
        {
            throw new InvalidDataException(index == ffmpeg.AVERROR_DECODER_NOT_FOUND
                ? $"The bundled FFmpeg has no decoder for the {kind} stream."
                : $"The file has no {kind} stream.");
        }

        AVStream* stream = _format->streams[index];
        _decoder = ffmpeg.avcodec_alloc_context3(codec);
        FfmpegError.CheckInput(ffmpeg.avcodec_parameters_to_context(_decoder, stream->codecpar), $"read the {kind} parameters");
        _decoder->thread_count = 0;
        _decoder->pkt_timebase = stream->time_base;
        FfmpegError.CheckInput(ffmpeg.avcodec_open2(_decoder, codec, null), $"open the {ToString(codec->name)} decoder");
        return stream;
    }

    private VideoFrame Convert(AVFrame* frame, VideoDecodeOptions options)
    {
        (int width, int height) = TargetSize(frame->width, frame->height, options.Width, options.Height);
        var source = (frame->width, frame->height, (AVPixelFormat)frame->format, frame->colorspace, frame->color_range);
        bool rgb = options.Format == VideoFrameFormat.Rgb24;
        if (_scale is null || _scaleSource != source)
        {
            if (_scale is not null)
            {
                ffmpeg.sws_freeContext(_scale);
            }

            AVPixelFormat target = rgb ? AVPixelFormat.AV_PIX_FMT_RGB24 : AVPixelFormat.AV_PIX_FMT_YUV420P;
            var flags = (int)(SwsFlags.SWS_BICUBIC | SwsFlags.SWS_ACCURATE_RND | SwsFlags.SWS_FULL_CHR_H_INT);
            _scale = ffmpeg.sws_getContext(frame->width, frame->height, source.Item3, width, height, target, flags, null, null, null);
            if (_scale is null)
            {
                throw new InvalidDataException($"FFmpeg cannot convert {ffmpeg.av_get_pix_fmt_name(source.Item3)} frames.");
            }

            int matrix = Matrix(frame->colorspace, frame->height);
            int_array4 sourceTable = Coefficients(matrix);
            int_array4 targetTable = Coefficients(rgb ? ffmpeg.SWS_CS_DEFAULT : matrix);
            int sourceRange = frame->color_range == AVColorRange.AVCOL_RANGE_JPEG ? 1 : 0;
            ffmpeg.sws_setColorspaceDetails(_scale, in sourceTable, sourceRange, in targetTable, rgb ? 1 : 0, 0, 1 << 16, 1 << 16);
            _scaleSource = source;
        }

        int chromaWidth = (width + 1) / 2;
        int chromaHeight = (height + 1) / 2;
        byte[] data = rgb ? new byte[width * height * 3] : new byte[width * height + 2 * chromaWidth * chromaHeight];
        fixed (byte* pixels = data)
        {
            byte*[] planes = rgb ? [pixels] : [pixels, pixels + width * height, pixels + width * height + chromaWidth * chromaHeight];
            int[] strides = rgb ? [width * 3] : [width, chromaWidth, chromaWidth];
            ffmpeg.sws_scale(_scale, frame->data, frame->linesize, 0, frame->height, planes, strides);
        }

        return new VideoFrame
        {
            Index = 0,
            Timestamp = TimeSpan.Zero,
            Width = width,
            Height = height,
            Format = options.Format,
            Data = data,
        };
    }

    private static void AppendConverted(SwrContext* resampler, byte** input, int count, int channels, List<float> samples)
    {
        int capacity = ffmpeg.swr_get_out_samples(resampler, count) + 32;
        if (capacity <= 0)
        {
            return;
        }

        float[] buffer = new float[capacity * channels];
        fixed (float* output = buffer)
        {
            byte* outputPlane = (byte*)output;
            int converted = ffmpeg.swr_convert(resampler, &outputPlane, capacity, input, count);
            if (converted > 0)
            {
                samples.AddRange(buffer.AsSpan(0, converted * channels));
            }
        }
    }

    private static (int Width, int Height) TargetSize(int width, int height, int? requestedWidth, int? requestedHeight)
    {
        return (requestedWidth, requestedHeight) switch
        {
            (int w, int h) => (w, h),
            (int w, null) => (w, Math.Max(1, (int)Math.Round((double)height * w / width))),
            (null, int h) => (Math.Max(1, (int)Math.Round((double)width * h / height)), h),
            _ => (width, height),
        };
    }

    private static int Matrix(AVColorSpace space, int height) => space switch
    {
        AVColorSpace.AVCOL_SPC_BT709 => ffmpeg.SWS_CS_ITU709,
        AVColorSpace.AVCOL_SPC_BT470BG or AVColorSpace.AVCOL_SPC_SMPTE170M => ffmpeg.SWS_CS_ITU601,
        AVColorSpace.AVCOL_SPC_BT2020_NCL or AVColorSpace.AVCOL_SPC_BT2020_CL => ffmpeg.SWS_CS_BT2020,
        AVColorSpace.AVCOL_SPC_SMPTE240M => ffmpeg.SWS_CS_SMPTE240M,
        AVColorSpace.AVCOL_SPC_FCC => ffmpeg.SWS_CS_FCC,
        _ => height >= 720 ? ffmpeg.SWS_CS_ITU709 : ffmpeg.SWS_CS_ITU601,
    };

    private static int_array4 Coefficients(int colorspace)
    {
        int* table = ffmpeg.sws_getCoefficients(colorspace);
        var result = new int_array4();
        for (uint i = 0; i < 4; i++)
        {
            result[i] = table[i];
        }

        return result;
    }

    private static double FrameRateOf(AVStream* stream)
    {
        double average = stream->avg_frame_rate.den != 0 ? ffmpeg.av_q2d(stream->avg_frame_rate) : 0;
        if (average > 0 && double.IsFinite(average))
        {
            return average;
        }

        double real = stream->r_frame_rate.den != 0 ? ffmpeg.av_q2d(stream->r_frame_rate) : 0;
        return real > 0 && double.IsFinite(real) ? real : 0;
    }

    private static string ToString(byte* text) => System.Runtime.InteropServices.Marshal.PtrToStringUTF8((nint)text) ?? string.Empty;
}
