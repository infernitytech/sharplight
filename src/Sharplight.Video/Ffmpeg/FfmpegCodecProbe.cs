using System.Collections.Concurrent;
using FFmpeg.AutoGen;

namespace Sharplight.Video.Ffmpeg;

/// <summary>
/// Which encoders work on this machine: software encoders when the build has them, hardware encoders only after a trial encode of a
/// few small frames succeeded (NVENC needs an NVIDIA driver, VideoToolbox and Media Foundation a usable encoder; some Media Foundation
/// encoders open and only fail on the first frame). Cached for the process.
/// </summary>
internal static unsafe class FfmpegCodecProbe
{
    private const int TrialFrames = 3;

    /// <summary>
    /// How long a hardware trial encode may take. Some drivers (Media Foundation encoders on machines without a usable device) block in
    /// open or drain instead of failing; such an encoder counts as unusable and its trial thread is abandoned.
    /// </summary>
    private static readonly TimeSpan s_trialTimeout = TimeSpan.FromSeconds(15);

    private static readonly ConcurrentDictionary<string, bool> s_usable = new(StringComparer.Ordinal);

    /// <summary>Whether the encoder of <paramref name="codec"/> can be used.</summary>
    public static bool IsUsable(FfmpegCodec codec) => s_usable.GetOrAdd(codec.EncoderName, _ => Probe(codec));

    /// <summary>The usable codecs of a format, the software ones first.</summary>
    /// <param name="container">The format.</param>
    /// <param name="includeHardware">Whether to probe and offer the hardware encoders.</param>
    public static IReadOnlyList<FfmpegCodec> Usable(FfmpegContainer container, bool includeHardware)
    {
        // Hardware encoders are not offered on Windows: Media Foundation encoders can block instead of failing, and the software codecs
        // cover what Sharplight needs there.
        bool hardware = includeHardware && !OperatingSystem.IsWindows();
        IEnumerable<FfmpegCodec> candidates = hardware ? container.Codecs.Concat(container.Hardware) : container.Codecs;
        return [.. candidates.Where(IsUsable)];
    }

    private static bool Probe(FfmpegCodec codec)
    {
        AVCodec* encoder = ffmpeg.avcodec_find_encoder_by_name(codec.EncoderName);
        if (encoder is null)
        {
            return false;
        }

        if (!codec.IsHardware)
        {
            return true;
        }

        bool usable = false;
        var trial = new Thread(() => usable = TrialEncode(codec))
        {
            IsBackground = true,
            Name = "Sharplight hardware encoder trial",
        };
        trial.Start();
        return trial.Join(s_trialTimeout) && usable;
    }

    private static bool TrialEncode(FfmpegCodec codec)
    {
        AVCodec* encoder = ffmpeg.avcodec_find_encoder_by_name(codec.EncoderName);
        AVCodecContext* context = ffmpeg.avcodec_alloc_context3(encoder);
        AVFrame* frame = ffmpeg.av_frame_alloc();
        AVPacket* packet = ffmpeg.av_packet_alloc();
        try
        {
            context->width = 256;
            context->height = 256;
            context->pix_fmt = ffmpeg.av_get_pix_fmt(codec.PixelFormats[0]);
            context->time_base = new AVRational { num = 1, den = 30 };
            context->framerate = new AVRational { num = 30, den = 1 };
            if (ffmpeg.avcodec_open2(context, encoder, null) < 0)
            {
                return false;
            }

            frame->format = (int)context->pix_fmt;
            frame->width = context->width;
            frame->height = context->height;
            if (ffmpeg.av_frame_get_buffer(frame, 0) < 0)
            {
                return false;
            }

            for (int i = 0; i < TrialFrames; i++)
            {
                frame->pts = i;
                if (!SendAndDrain(context, frame, packet))
                {
                    return false;
                }
            }

            return SendAndDrain(context, null, packet);
        }
        finally
        {
            ffmpeg.av_packet_free(&packet);
            ffmpeg.av_frame_free(&frame);
            ffmpeg.avcodec_free_context(&context);
        }
    }

    private static bool SendAndDrain(AVCodecContext* context, AVFrame* frame, AVPacket* packet)
    {
        if (ffmpeg.avcodec_send_frame(context, frame) < 0)
        {
            return false;
        }

        while (true)
        {
            int result = ffmpeg.avcodec_receive_packet(context, packet);
            if (FfmpegError.IsAgainOrEnd(result))
            {
                return true;
            }

            if (result < 0)
            {
                return false;
            }

            ffmpeg.av_packet_unref(packet);
        }
    }
}
