using System.Collections.Concurrent;
using FFmpeg.AutoGen;

namespace Sharplight.Video.Ffmpeg;

/// <summary>
/// Which encoders work on this machine: software encoders when the build has them, hardware encoders only after a trial open of a small
/// frame succeeded (NVENC needs an NVIDIA driver, VideoToolbox and Media Foundation a usable encoder). Cached for the process.
/// </summary>
internal static unsafe class FfmpegCodecProbe
{
    private static readonly ConcurrentDictionary<string, bool> s_usable = new(StringComparer.Ordinal);

    /// <summary>Whether the encoder of <paramref name="codec"/> can be used.</summary>
    public static bool IsUsable(FfmpegCodec codec) => s_usable.GetOrAdd(codec.EncoderName, _ => Probe(codec));

    /// <summary>The usable codecs of a format, the software ones first.</summary>
    /// <param name="container">The format.</param>
    /// <param name="includeHardware">Whether to probe and offer the hardware encoders.</param>
    public static IReadOnlyList<FfmpegCodec> Usable(FfmpegContainer container, bool includeHardware)
    {
        IEnumerable<FfmpegCodec> candidates = includeHardware ? container.Codecs.Concat(container.Hardware) : container.Codecs;
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

        AVCodecContext* context = ffmpeg.avcodec_alloc_context3(encoder);
        try
        {
            context->width = 256;
            context->height = 256;
            context->pix_fmt = ffmpeg.av_get_pix_fmt(codec.PixelFormats[0]);
            context->time_base = new AVRational { num = 1, den = 30 };
            context->framerate = new AVRational { num = 30, den = 1 };
            return ffmpeg.avcodec_open2(context, encoder, null) >= 0;
        }
        finally
        {
            ffmpeg.avcodec_free_context(&context);
        }
    }
}
