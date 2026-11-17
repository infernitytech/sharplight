namespace Sharplight.Video.Ffmpeg;

/// <summary>The output formats of the bundled FFmpeg encoders.</summary>
internal static class FfmpegContainers
{
    public static readonly FfmpegContainer Mp4 = new()
    {
        Format = VideoFormats.Mp4,
        FileExtensions = [".mp4", ".m4v"],
        Muxer = "mp4",
        Codecs = [FfmpegCodecs.H264, FfmpegCodecs.Av1, FfmpegCodecs.Vp9],
        Hardware = [FfmpegCodecs.H264Nvenc, FfmpegCodecs.HevcNvenc, FfmpegCodecs.Av1Nvenc, FfmpegCodecs.H264VideoToolbox, FfmpegCodecs.HevcVideoToolbox,
            FfmpegCodecs.H264MediaFoundation, FfmpegCodecs.HevcMediaFoundation],
        AudioCodecs = [VideoCodecs.Aac, VideoCodecs.Opus],
    };

    public static readonly FfmpegContainer WebM = new()
    {
        Format = VideoFormats.WebM,
        FileExtensions = [".webm"],
        Muxer = "webm",
        Codecs = [FfmpegCodecs.Vp9, FfmpegCodecs.Av1, FfmpegCodecs.Vp8],
        Hardware = [FfmpegCodecs.Av1Nvenc],
        AudioCodecs = [VideoCodecs.Opus],
    };

    public static readonly FfmpegContainer Mov = new()
    {
        Format = VideoFormats.Mov,
        FileExtensions = [".mov"],
        Muxer = "mov",
        Codecs = [FfmpegCodecs.ProRes, FfmpegCodecs.H264, FfmpegCodecs.Ffv1],
        Hardware = [FfmpegCodecs.HevcVideoToolbox, FfmpegCodecs.H264VideoToolbox, FfmpegCodecs.HevcNvenc, FfmpegCodecs.H264Nvenc],
        AudioCodecs = [VideoCodecs.Aac, VideoCodecs.Pcm],
    };

    public static readonly FfmpegContainer Matroska = new()
    {
        Format = VideoFormats.Matroska,
        FileExtensions = [".mkv"],
        Muxer = "matroska",
        Codecs = [FfmpegCodecs.H264, FfmpegCodecs.Av1, FfmpegCodecs.Vp9, FfmpegCodecs.Ffv1, FfmpegCodecs.ProRes],
        Hardware = [FfmpegCodecs.HevcNvenc, FfmpegCodecs.Av1Nvenc, FfmpegCodecs.H264Nvenc, FfmpegCodecs.HevcVideoToolbox, FfmpegCodecs.H264VideoToolbox,
            FfmpegCodecs.HevcMediaFoundation, FfmpegCodecs.H264MediaFoundation],
        AudioCodecs = [VideoCodecs.Opus, VideoCodecs.Aac, VideoCodecs.Flac],
    };

    public static readonly FfmpegContainer Gif = new()
    {
        Format = VideoFormats.Gif,
        FileExtensions = [".gif"],
        Muxer = "gif",
        Codecs = [FfmpegCodecs.Gif],
    };

    public static readonly FfmpegContainer Apng = new()
    {
        Format = VideoFormats.Apng,
        FileExtensions = [".apng", ".png"],
        Muxer = "apng",
        Codecs = [FfmpegCodecs.Apng],
    };

    /// <summary>Every format, in registration order.</summary>
    public static readonly FfmpegContainer[] All = [Mp4, WebM, Mov, Matroska, Gif, Apng];

    /// <summary>The container of a format id.</summary>
    /// <exception cref="ArgumentException">The format is not one of the bundled formats.</exception>
    public static FfmpegContainer ForFormat(string format) =>
        All.FirstOrDefault(c => string.Equals(c.Format, format, StringComparison.OrdinalIgnoreCase))
        ?? throw new ArgumentException($"'{format}' is not a format of the bundled FFmpeg encoders ({string.Join(", ", All.Select(c => c.Format))}).", nameof(format));
}
