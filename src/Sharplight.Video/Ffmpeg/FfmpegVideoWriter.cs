using FFmpeg.AutoGen;
using Sharplight.Audio;
using Sharplight.Vision;

namespace Sharplight.Video.Ffmpeg;

/// <summary>
/// Encodes frames with the bundled FFmpeg into a stream as they come: RGB frames are converted by swscale (BT.709, limited range, tagged
/// in the stream) or, for GIF, run through a palettegen / paletteuse filter graph (which holds every frame until <see cref="Finish"/>).
/// The optional sound track is a clip's <see cref="AudioBuffer"/> (resampled by swresample), or a source file's audio stream, copied
/// packet by packet when the muxer accepts its codec and transcoded otherwise; either way it is interleaved with the video by time and
/// ends with it. One instance per encode; not thread safe.
/// </summary>
internal sealed unsafe class FfmpegVideoWriter : IDisposable
{
    private readonly FfmpegEncodePlan _plan;
    private readonly FfmpegMuxer _muxer;
    private AVCodecContext* _video;
    private AVStream* _videoStream;
    private SwsContext* _scale;
    private AVFrame* _frame;
    private PaletteFilter? _palette;
    private byte[]? _padded;
    private FfmpegAudioEncoder? _audio;
    private FfmpegAudioInput? _audioInput;
    private FfmpegAudioCopy? _audioCopy;
    private FfmpegAudioDecoder? _audioDecoder;
    private long _audioTotal = long.MaxValue;
    private bool _finished;

    private FfmpegVideoWriter(FfmpegEncodePlan plan, Stream output)
    {
        _plan = plan;
        _muxer = new FfmpegMuxer(plan.Container.Muxer, output);
        try
        {
            OpenVideo();
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The frames added so far.</summary>
    public long FramesWritten { get; private set; }

    /// <summary>Encodes a whole clip with its <see cref="VideoClip.Audio"/> (cut or padded with silence to the clip's length).</summary>
    /// <param name="plan">The resolved settings.</param>
    /// <param name="clip">The clip.</param>
    /// <param name="output">The destination.</param>
    /// <param name="cancellationToken">Checked between frames.</param>
    public static void Write(FfmpegEncodePlan plan, VideoClip clip, Stream output, CancellationToken cancellationToken)
    {
        using var writer = new FfmpegVideoWriter(plan, output);
        if (plan.Audio is not null && clip.Audio is not null)
        {
            writer.OpenAudio(plan.Audio, clip.Audio, clip.Frames.Count);
        }

        writer.Start();
        foreach (ImageInput frame in clip.Frames)
        {
            cancellationToken.ThrowIfCancellationRequested();
            writer.Add(frame);
        }

        writer.Finish();
    }

    /// <summary>Starts an incremental encode whose sound track comes from a media file.</summary>
    /// <param name="plan">The resolved settings (<see cref="FfmpegEncodePlan.Audio"/> null: no sound).</param>
    /// <param name="output">The destination.</param>
    /// <param name="audioSource">The media file with the sound, or <see langword="null"/>.</param>
    /// <param name="audioStart">Where the sound starts in the file.</param>
    /// <returns>The writer, header written.</returns>
    public static FfmpegVideoWriter Open(FfmpegEncodePlan plan, Stream output, string? audioSource, TimeSpan audioStart)
    {
        var writer = new FfmpegVideoWriter(plan, output);
        try
        {
            if (plan.Audio is not null && audioSource is not null)
            {
                writer.OpenAudio(plan.Audio, audioSource, audioStart);
            }

            writer.Start();
            return writer;
        }
        catch
        {
            writer.Dispose();
            throw;
        }
    }

    /// <summary>Encodes the next frame (padded to the encoded size when the codec needs even sizes).</summary>
    /// <param name="image">The frame, of the size the plan was made for.</param>
    public void Add(ImageInput image)
    {
        WriteAudioUntil(FramesWritten, final: false);
        if (_padded is null && (image.Width != _plan.Width || image.Height != _plan.Height))
        {
            _padded = new byte[_plan.Width * _plan.Height * 3];
        }

        byte[] rgb = _padded is null ? image.Rgb : Pad(image, _padded);
        if (_palette is not null)
        {
            _palette.Add(rgb, FramesWritten);
        }
        else
        {
            EncodeVideo(rgb, FramesWritten);
        }

        FramesWritten++;
    }

    /// <summary>Flushes the encoders, writes the sound up to the end of the video and the trailer.</summary>
    public void Finish()
    {
        if (_finished)
        {
            return;
        }

        _finished = true;
        if (_palette is not null)
        {
            _palette.Finish(frame => _muxer.Send(_video, _videoStream, frame));
        }

        _muxer.Send(_video, _videoStream, null);
        WriteAudioUntil(FramesWritten, final: true);
        _audio?.Flush();
        _muxer.Finish();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _palette?.Dispose();
        _palette = null;
        if (_scale is not null)
        {
            ffmpeg.sws_freeContext(_scale);
            _scale = null;
        }

        _audioDecoder?.Dispose();
        _audioCopy?.Dispose();
        _audio?.Dispose();
        _audioInput?.Dispose();
        _audioDecoder = null;
        _audioCopy = null;
        _audio = null;
        _audioInput = null;
        AVFrame* frame = _frame;
        ffmpeg.av_frame_free(&frame);
        _frame = null;
        AVCodecContext* video = _video;
        ffmpeg.avcodec_free_context(&video);
        _video = null;
        _muxer.Dispose();
    }

    /// <summary>Writes the file header once every stream is added.</summary>
    private void Start()
    {
        AVDictionary* muxerOptions = Dictionary(_plan.MuxerOptions);
        try
        {
            _muxer.Check(ffmpeg.avformat_write_header(_muxer.Format, &muxerOptions), "write the file header");
        }
        finally
        {
            ffmpeg.av_dict_free(&muxerOptions);
        }
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

        if (_muxer.NeedsGlobalHeader)
        {
            _video->flags |= ffmpeg.AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        AVDictionary* options = Dictionary(_plan.EncoderOptions);
        try
        {
            _muxer.Check(ffmpeg.avcodec_open2(_video, codec, &options), $"open the {_plan.Codec.EncoderName} encoder");
        }
        finally
        {
            ffmpeg.av_dict_free(&options);
        }

        _videoStream = ffmpeg.avformat_new_stream(_muxer.Format, null);
        _muxer.Check(ffmpeg.avcodec_parameters_from_context(_videoStream->codecpar, _video), "describe the video stream");
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
        _muxer.Check(ffmpeg.av_frame_get_buffer(_frame, 0), "allocate a video frame");
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


    private void OpenAudio(FfmpegAudioPlan plan, AudioBuffer audio, long frames)
    {
        _audio = new FfmpegAudioEncoder(_muxer, plan, audio.SampleRate, audio.Channels);
        // The track has the clip's length: the samples are cut or padded with silence after resampling.
        _audioTotal = (long)Math.Round(frames * (double)_plan.RateDenominator / _plan.RateNumerator * _audio.SampleRate);
        SwrContext* resampler = null;
        AVChannelLayout input = default;
        ffmpeg.av_channel_layout_default(&input, audio.Channels);
        try
        {
            _muxer.Check(ffmpeg.swr_alloc_set_opts2(&resampler, _audio.Layout, _audio.SampleFormat, _audio.SampleRate, &input, AVSampleFormat.AV_SAMPLE_FMT_FLT, audio.SampleRate, 0, null), "set up the resampler");
            _muxer.Check(ffmpeg.swr_init(resampler), "set up the resampler");
            int capacity = ffmpeg.swr_get_out_samples(resampler, audio.FrameCount) + 256;
            byte** output = stackalloc byte*[8];
            int lineSize;
            _muxer.Check(ffmpeg.av_samples_alloc(output, &lineSize, _audio.Layout->nb_channels, capacity, _audio.SampleFormat, 0), "allocate audio samples");
            try
            {
                fixed (float* samples = audio.Samples)
                {
                    byte* inputData = (byte*)samples;
                    int converted = _muxer.Check(ffmpeg.swr_convert(resampler, output, capacity, &inputData, audio.FrameCount), "resample the audio");
                    _audio.Append(output, converted, _audioTotal);
                    converted = _muxer.Check(ffmpeg.swr_convert(resampler, output, capacity, null, 0), "resample the audio");
                    _audio.Append(output, converted, _audioTotal);
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

    private void OpenAudio(FfmpegAudioPlan plan, string path, TimeSpan start)
    {
        _audioInput = FfmpegAudioInput.Open(path, start);
        if (_audioInput is null)
        {
            return;
        }

        AVCodecParameters* source = _audioInput.Stream->codecpar;
        if (!plan.CodecRequested && _muxer.Accepts(source->codec_id))
        {
            _audioCopy = new FfmpegAudioCopy(_muxer, _audioInput, skipBeforeStart: start > TimeSpan.Zero);
            return;
        }

        _audio = new FfmpegAudioEncoder(_muxer, plan, source->sample_rate, source->ch_layout.nb_channels);
        _audioDecoder = new FfmpegAudioDecoder(_audioInput, _audio);
    }

    /// <summary>Writes the sound up to the start of video frame <paramref name="frameIndex"/>; <paramref name="final"/>: to the video's end, which ends the track.</summary>
    private void WriteAudioUntil(long frameIndex, bool final)
    {
        double seconds = frameIndex * (double)_plan.RateDenominator / _plan.RateNumerator;
        _audioCopy?.WriteUntil(seconds);
        if (_audio is null)
        {
            return;
        }

        long until = (long)(seconds * _audio.SampleRate);
        if (_audioDecoder is null)
        {
            _audio.SendUntil(final ? _audioTotal : Math.Min(_audioTotal, until), _audioTotal, final);
            return;
        }

        if (final)
        {
            _audioDecoder.FillTo(until);
            long total = Math.Min(_audio.Available, until);
            _audio.SendUntil(total, total, final: true);
            return;
        }

        _audioDecoder.FillTo(until + _audio.FrameSize);
        _audio.SendUntil(until, _audioDecoder.Ended ? _audio.Available : long.MaxValue, final: false);
    }

    private void EncodeVideo(byte[] rgb, long index)
    {
        _muxer.Check(ffmpeg.av_frame_make_writable(_frame), "reuse the video frame");
        fixed (byte* pixels = rgb)
        {
            byte*[] source = [pixels];
            int[] stride = [_plan.Width * 3];
            ffmpeg.sws_scale(_scale, source, stride, 0, _plan.Height, _frame->data, _frame->linesize);
        }

        _frame->pts = index;
        _muxer.Send(_video, _videoStream, _frame);
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
