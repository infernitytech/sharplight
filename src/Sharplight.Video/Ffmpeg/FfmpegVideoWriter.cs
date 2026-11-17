using FFmpeg.AutoGen;
using Sharplight.Audio;
using Sharplight.Vision;

namespace Sharplight.Video.Ffmpeg;

/// <summary>
/// Encodes a <see cref="VideoClip"/> with the bundled FFmpeg into a stream: RGB frames are converted by swscale (BT.709, limited range,
/// tagged in the stream) or, for GIF, run through a palettegen / paletteuse filter graph; the optional audio track is resampled by
/// swresample and interleaved by time. One instance per encode; not thread safe.
/// </summary>
internal sealed unsafe class FfmpegVideoWriter : IDisposable
{
    private readonly FfmpegEncodePlan _plan;
    private readonly VideoClip _clip;
    private readonly OutputStreamIo _io;
    private AVFormatContext* _format;
    private AVCodecContext* _video;
    private AVStream* _videoStream;
    private AVCodecContext* _audio;
    private AVStream* _audioStream;
    private AVAudioFifo* _audioFifo;
    private SwsContext* _scale;
    private AVFrame* _frame;
    private AVFrame* _audioFrame;
    private AVPacket* _packet;
    private PaletteFilter? _palette;
    private long _audioSamplesSent;
    private long _audioSamplesTotal;

    private FfmpegVideoWriter(FfmpegEncodePlan plan, VideoClip clip, Stream output)
    {
        _plan = plan;
        _clip = clip;
        _io = new OutputStreamIo(output);
    }

    /// <summary>Encodes the whole clip.</summary>
    /// <param name="plan">The resolved settings.</param>
    /// <param name="clip">The clip.</param>
    /// <param name="output">The destination.</param>
    /// <param name="cancellationToken">Checked between frames.</param>
    public static void Write(FfmpegEncodePlan plan, VideoClip clip, Stream output, CancellationToken cancellationToken)
    {
        using var writer = new FfmpegVideoWriter(plan, clip, output);
        writer.Run(cancellationToken);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _palette?.Dispose();
        if (_scale is not null)
        {
            ffmpeg.sws_freeContext(_scale);
            _scale = null;
        }

        if (_audioFifo is not null)
        {
            ffmpeg.av_audio_fifo_free(_audioFifo);
            _audioFifo = null;
        }

        AVFrame* frame = _frame;
        ffmpeg.av_frame_free(&frame);
        AVFrame* audioFrame = _audioFrame;
        ffmpeg.av_frame_free(&audioFrame);
        AVPacket* packet = _packet;
        ffmpeg.av_packet_free(&packet);
        AVCodecContext* video = _video;
        ffmpeg.avcodec_free_context(&video);
        AVCodecContext* audio = _audio;
        ffmpeg.avcodec_free_context(&audio);
        _frame = null;
        _audioFrame = null;
        _packet = null;
        _video = null;
        _audio = null;
        if (_format is not null)
        {
            _format->pb = null;
            ffmpeg.avformat_free_context(_format);
            _format = null;
        }

        _io.Dispose();
    }

    private void Run(CancellationToken cancellationToken)
    {
        AVFormatContext* format = null;
        Check(ffmpeg.avformat_alloc_output_context2(&format, null, _plan.Container.Muxer, null), $"create the {_plan.Container.Muxer} muxer");
        _format = format;
        _format->pb = _io.Context;
        _packet = ffmpeg.av_packet_alloc();
        OpenVideo();
        if (_plan.Audio is not null && _clip.Audio is not null)
        {
            OpenAudio(_plan.Audio, _clip.Audio);
        }

        AVDictionary* muxerOptions = Dictionary(_plan.MuxerOptions);
        try
        {
            Check(ffmpeg.avformat_write_header(_format, &muxerOptions), "write the file header");
        }
        finally
        {
            ffmpeg.av_dict_free(&muxerOptions);
        }

        byte[]? padded = _plan.Width != _clip.Width || _plan.Height != _clip.Height ? new byte[_plan.Width * _plan.Height * 3] : null;
        for (int i = 0; i < _clip.Frames.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            WriteAudioUntil(i);
            ImageInput image = _clip.Frames[i];
            byte[] rgb = padded is null ? image.Rgb : Pad(image, padded);
            if (_palette is not null)
            {
                _palette.Add(rgb, i);
            }
            else
            {
                EncodeVideo(rgb, i);
            }
        }

        if (_palette is not null)
        {
            _palette.Finish(frame => SendVideo(frame));
        }

        SendVideo(null);
        WriteAudioUntil(long.MaxValue);
        if (_audio is not null)
        {
            Send(_audio, _audioStream, null);
        }

        Check(ffmpeg.av_write_trailer(_format), "finish the file");
        ffmpeg.avio_flush(_io.Context);
        ThrowIfOutputFailed();
    }

    private void OpenVideo()
    {
        AVCodec* codec = ffmpeg.avcodec_find_encoder_by_name(_plan.Codec.EncoderName);
        if (codec is null)
        {
            throw new PlatformNotSupportedException($"The bundled FFmpeg has no {_plan.Codec.EncoderName} encoder.");
        }

        AVPixelFormat pixelFormat = ffmpeg.av_get_pix_fmt(_plan.PixelFormat);
        _video = ffmpeg.avcodec_alloc_context3(codec);
        _video->width = _plan.Width;
        _video->height = _plan.Height;
        _video->pix_fmt = pixelFormat;
        _video->time_base = new AVRational { num = _plan.RateDenominator, den = _plan.RateNumerator };
        _video->framerate = new AVRational { num = _plan.RateNumerator, den = _plan.RateDenominator };
        _video->sample_aspect_ratio = new AVRational { num = 1, den = 1 };
        _video->thread_count = _plan.Threads;
        if (_plan.KeyframeInterval is int interval)
        {
            _video->gop_size = interval;
            _video->keyint_min = Math.Min(interval, _video->keyint_min);
        }

        if (_plan.BitRate > 0)
        {
            _video->bit_rate = _plan.BitRate;
        }

        if (_plan.QuantizerMinimum is int qmin && _plan.QuantizerMaximum is int qmax)
        {
            _video->qmin = qmin;
            _video->qmax = qmax;
        }

        if (_plan.GlobalQualityPercent is int percent)
        {
            _video->flags |= ffmpeg.AV_CODEC_FLAG_QSCALE;
            _video->global_quality = percent * ffmpeg.FF_QP2LAMBDA;
        }

        if (_plan.CompressionLevel is int level)
        {
            _video->compression_level = level;
        }

        bool isYuv = _plan.PixelFormat.StartsWith("yuv", StringComparison.Ordinal) || _plan.PixelFormat is "nv12" or "p010le";
        if (isYuv)
        {
            _video->colorspace = AVColorSpace.AVCOL_SPC_BT709;
            _video->color_primaries = AVColorPrimaries.AVCOL_PRI_BT709;
            _video->color_trc = AVColorTransferCharacteristic.AVCOL_TRC_BT709;
            _video->color_range = AVColorRange.AVCOL_RANGE_MPEG;
        }
        else if (_plan.PixelFormat is "gbrp" or "bgr0")
        {
            _video->colorspace = AVColorSpace.AVCOL_SPC_RGB;
        }

        if ((_format->oformat->flags & ffmpeg.AVFMT_GLOBALHEADER) != 0)
        {
            _video->flags |= ffmpeg.AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        AVDictionary* options = Dictionary(_plan.EncoderOptions);
        try
        {
            Check(ffmpeg.avcodec_open2(_video, codec, &options), $"open the {_plan.Codec.EncoderName} encoder");
        }
        finally
        {
            ffmpeg.av_dict_free(&options);
        }

        _videoStream = ffmpeg.avformat_new_stream(_format, null);
        Check(ffmpeg.avcodec_parameters_from_context(_videoStream->codecpar, _video), "describe the video stream");
        _videoStream->time_base = _video->time_base;
        _videoStream->avg_frame_rate = _video->framerate;
        _frame = ffmpeg.av_frame_alloc();
        if (_plan.Codec.UsesPaletteFilter)
        {
            _palette = new PaletteFilter(_plan.Width, _plan.Height, _video->time_base, _plan.Dither);
            return;
        }

        _frame->format = (int)pixelFormat;
        _frame->width = _plan.Width;
        _frame->height = _plan.Height;
        Check(ffmpeg.av_frame_get_buffer(_frame, 0), "allocate a video frame");
        var flags = (int)(SwsFlags.SWS_BICUBIC | SwsFlags.SWS_ACCURATE_RND | SwsFlags.SWS_FULL_CHR_H_INT);
        _scale = ffmpeg.sws_getContext(_plan.Width, _plan.Height, AVPixelFormat.AV_PIX_FMT_RGB24, _plan.Width, _plan.Height, pixelFormat, flags, null, null, null);
        if (_scale is null)
        {
            throw new InvalidOperationException($"FFmpeg cannot convert RGB to {_plan.PixelFormat}.");
        }

        if (isYuv)
        {
            int_array4 source = Coefficients(ffmpeg.SWS_CS_DEFAULT);
            int_array4 destination = Coefficients(ffmpeg.SWS_CS_ITU709);
            ffmpeg.sws_setColorspaceDetails(_scale, in source, 1, in destination, 0, 0, 1 << 16, 1 << 16);
        }
    }

    private void OpenAudio(FfmpegAudioPlan plan, AudioBuffer audio)
    {
        AVCodec* codec = ffmpeg.avcodec_find_encoder_by_name(plan.EncoderName);
        if (codec is null)
        {
            throw new PlatformNotSupportedException($"The bundled FFmpeg has no {plan.EncoderName} encoder.");
        }

        _audio = ffmpeg.avcodec_alloc_context3(codec);
        int channels = Math.Min(audio.Channels, 2);
        ffmpeg.av_channel_layout_default(&_audio->ch_layout, channels);
        _audio->sample_rate = PickSampleRate(codec, audio.SampleRate);
        _audio->sample_fmt = PickSampleFormat(codec);
        _audio->time_base = new AVRational { num = 1, den = _audio->sample_rate };
        if (plan.BitRate > 0)
        {
            _audio->bit_rate = plan.BitRate;
        }

        if ((_format->oformat->flags & ffmpeg.AVFMT_GLOBALHEADER) != 0)
        {
            _audio->flags |= ffmpeg.AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        Check(ffmpeg.avcodec_open2(_audio, codec, null), $"open the {plan.EncoderName} encoder");
        _audioStream = ffmpeg.avformat_new_stream(_format, null);
        Check(ffmpeg.avcodec_parameters_from_context(_audioStream->codecpar, _audio), "describe the audio stream");
        _audioStream->time_base = _audio->time_base;

        // The track has the clip's length: the samples are cut or padded with silence after resampling.
        _audioSamplesTotal = (long)Math.Round(_clip.Frames.Count * (double)_plan.RateDenominator / _plan.RateNumerator * _audio->sample_rate);
        _audioFifo = ffmpeg.av_audio_fifo_alloc(_audio->sample_fmt, channels, (int)Math.Max(1, _audioSamplesTotal));
        Resample(audio, channels);
        int frameSize = _audio->frame_size > 0 ? _audio->frame_size : 1024;
        _audioFrame = ffmpeg.av_frame_alloc();
        _audioFrame->format = (int)_audio->sample_fmt;
        _audioFrame->sample_rate = _audio->sample_rate;
        _audioFrame->nb_samples = frameSize;
        Check(ffmpeg.av_channel_layout_copy(&_audioFrame->ch_layout, &_audio->ch_layout), "set the channel layout");
        Check(ffmpeg.av_frame_get_buffer(_audioFrame, 0), "allocate an audio frame");
    }

    private void Resample(AudioBuffer audio, int channels)
    {
        SwrContext* resampler = null;
        AVChannelLayout input = default;
        ffmpeg.av_channel_layout_default(&input, audio.Channels);
        try
        {
            Check(ffmpeg.swr_alloc_set_opts2(&resampler, &_audio->ch_layout, _audio->sample_fmt, _audio->sample_rate, &input, AVSampleFormat.AV_SAMPLE_FMT_FLT, audio.SampleRate, 0, null), "set up the resampler");
            Check(ffmpeg.swr_init(resampler), "set up the resampler");
            int capacity = ffmpeg.swr_get_out_samples(resampler, audio.FrameCount) + 256;
            byte** output = stackalloc byte*[8];
            int lineSize;
            Check(ffmpeg.av_samples_alloc(output, &lineSize, channels, capacity, _audio->sample_fmt, 0), "allocate audio samples");
            try
            {
                fixed (float* samples = audio.Samples)
                {
                    byte* inputData = (byte*)samples;
                    int converted = Check(ffmpeg.swr_convert(resampler, output, capacity, &inputData, audio.FrameCount), "resample the audio");
                    AppendToFifo(output, converted);
                    converted = Check(ffmpeg.swr_convert(resampler, output, capacity, null, 0), "resample the audio");
                    AppendToFifo(output, converted);
                }
            }
            finally
            {
                ffmpeg.av_freep(&output[0]);
            }
        }
        finally
        {
            ffmpeg.swr_free(&resampler);
            ffmpeg.av_channel_layout_uninit(&input);
        }
    }

    private void AppendToFifo(byte** samples, int count)
    {
        long room = _audioSamplesTotal - ffmpeg.av_audio_fifo_size(_audioFifo);
        int take = (int)Math.Clamp(room, 0, count);
        if (take > 0)
        {
            Check(ffmpeg.av_audio_fifo_write(_audioFifo, (void**)samples, take), "buffer the audio");
        }
    }

    /// <summary>Sends the audio up to the start of video frame <paramref name="frameIndex"/> (all of it for <see cref="long.MaxValue"/>).</summary>
    private void WriteAudioUntil(long frameIndex)
    {
        if (_audio is null)
        {
            return;
        }

        double seconds = frameIndex == long.MaxValue ? double.MaxValue : frameIndex * (double)_plan.RateDenominator / _plan.RateNumerator;
        long until = seconds >= double.MaxValue / 2 ? _audioSamplesTotal : Math.Min(_audioSamplesTotal, (long)(seconds * _audio->sample_rate));
        int frameSize = _audio->frame_size > 0 ? _audio->frame_size : 1024;
        bool variable = (_audio->codec->capabilities & (ffmpeg.AV_CODEC_CAP_VARIABLE_FRAME_SIZE | ffmpeg.AV_CODEC_CAP_SMALL_LAST_FRAME)) != 0;
        while (_audioSamplesSent < until)
        {
            long remaining = _audioSamplesTotal - _audioSamplesSent;
            if (remaining < frameSize && frameIndex != long.MaxValue)
            {
                return;
            }

            Check(ffmpeg.av_frame_make_writable(_audioFrame), "reuse the audio frame");
            int count = (int)Math.Min(frameSize, remaining);
            void** data = (void**)&_audioFrame->data;
            int read = ffmpeg.av_audio_fifo_read(_audioFifo, data, count);
            int samples = Math.Max(read, 0);
            if (samples < count)
            {
                ffmpeg.av_samples_set_silence((byte**)data, samples, count - samples, _audio->ch_layout.nb_channels, _audio->sample_fmt);
            }

            int sent = variable ? count : frameSize;
            if (sent > count)
            {
                ffmpeg.av_samples_set_silence((byte**)data, count, sent - count, _audio->ch_layout.nb_channels, _audio->sample_fmt);
            }

            _audioFrame->nb_samples = sent;
            _audioFrame->pts = _audioSamplesSent;
            _audioSamplesSent += count;
            Send(_audio, _audioStream, _audioFrame);
        }
    }

    private void EncodeVideo(byte[] rgb, long index)
    {
        Check(ffmpeg.av_frame_make_writable(_frame), "reuse the video frame");
        fixed (byte* pixels = rgb)
        {
            byte*[] source = [pixels];
            int[] stride = [_plan.Width * 3];
            ffmpeg.sws_scale(_scale, source, stride, 0, _plan.Height, _frame->data, _frame->linesize);
        }

        _frame->pts = index;
        SendVideo(_frame);
    }

    private void SendVideo(AVFrame* frame)
    {
        Send(_video, _videoStream, frame);
    }

    private void Send(AVCodecContext* encoder, AVStream* stream, AVFrame* frame)
    {
        Check(ffmpeg.avcodec_send_frame(encoder, frame), "encode a frame");
        while (true)
        {
            int result = ffmpeg.avcodec_receive_packet(encoder, _packet);
            if (FfmpegError.IsAgainOrEnd(result))
            {
                return;
            }

            Check(result, "encode a frame");
            ffmpeg.av_packet_rescale_ts(_packet, encoder->time_base, stream->time_base);
            _packet->stream_index = stream->index;
            Check(ffmpeg.av_interleaved_write_frame(_format, _packet), "write a packet");
        }
    }

    private int Check(int result, string action)
    {
        if (result < 0)
        {
            ThrowIfOutputFailed();
        }

        return FfmpegError.Check(result, action);
    }

    private void ThrowIfOutputFailed()
    {
        if (_io.Failure is not null)
        {
            throw new IOException($"Writing the video failed: {_io.Failure.Message}", _io.Failure);
        }
    }

    /// <summary>Copies the image into the even-sized buffer, repeating the last column and row.</summary>
    private byte[] Pad(ImageInput image, byte[] padded)
    {
        int sourceStride = image.Width * 3;
        int stride = _plan.Width * 3;
        for (int y = 0; y < _plan.Height; y++)
        {
            int sourceRow = Math.Min(y, image.Height - 1) * sourceStride;
            Buffer.BlockCopy(image.Rgb, sourceRow, padded, y * stride, sourceStride);
            for (int x = image.Width; x < _plan.Width; x++)
            {
                Buffer.BlockCopy(image.Rgb, sourceRow + sourceStride - 3, padded, y * stride + x * 3, 3);
            }
        }

        return padded;
    }

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

    private static AVDictionary* Dictionary(IReadOnlyDictionary<string, string> values)
    {
        AVDictionary* dictionary = null;
        foreach ((string key, string value) in values)
        {
            ffmpeg.av_dict_set(&dictionary, key, value, 0);
        }

        return dictionary;
    }
}
