namespace Sharplight.Video.Ffmpeg;

/// <summary>The codecs of Sharplight's FFmpeg build (native/ffmpeg/build.sh enables exactly these encoders).</summary>
internal static class FfmpegCodecs
{
    public static readonly FfmpegCodec H264 = new()
    {
        Id = VideoCodecs.H264,
        DisplayName = "H.264",
        Description = "H.264 (OpenH264): plays on every device and browser",
        EncoderName = "libopenh264",
        PixelFormats = ["yuv420p"],
        Quality = QualityControl.QuantizerWindow,
        WorstRateFactor = 40,
        BestRateFactor = 16,
        RateFactorRange = (1, 51),
        FixedOptions = Options(("profile", "high"), ("coder", "cabac")),
    };

    public static readonly FfmpegCodec Av1 = new()
    {
        Id = VideoCodecs.Av1,
        DisplayName = "AV1",
        Description = "AV1 (SVT-AV1): the smallest files; plays in current browsers and players",
        EncoderName = "libsvtav1",
        PixelFormats = ["yuv420p", "yuv420p10le"],
        Presets = Presets(("preset", ["12", "10", "8", "6", "4"])),
        DefaultPreset = VideoPresets.Medium,
        Quality = QualityControl.Crf,
        WorstRateFactor = 55,
        BestRateFactor = 15,
        RateFactorRange = (1, 63),
    };

    public static readonly FfmpegCodec Vp9 = new()
    {
        Id = VideoCodecs.Vp9,
        DisplayName = "VP9",
        Description = "VP9 (libvpx): royalty-free, plays in every browser",
        EncoderName = "libvpx-vp9",
        PixelFormats = ["yuv420p", "yuv444p", "yuv420p10le"],
        Presets = VpxPresets(8),
        DefaultPreset = VideoPresets.Fast,
        Quality = QualityControl.Crf,
        WorstRateFactor = 55,
        BestRateFactor = 15,
        RateFactorRange = (0, 63),
        FixedOptions = Options(("row-mt", "1"), ("b", "0")),
        NeedsEvenSize = false,
    };

    public static readonly FfmpegCodec Vp8 = new()
    {
        Id = VideoCodecs.Vp8,
        DisplayName = "VP8",
        Description = "VP8 (libvpx): for older WebM players",
        EncoderName = "libvpx",
        PixelFormats = ["yuv420p"],
        Presets = VpxPresets(16),
        DefaultPreset = VideoPresets.Fast,
        Quality = QualityControl.Crf,
        WorstRateFactor = 50,
        BestRateFactor = 6,
        RateFactorRange = (4, 63),
        NeedsBitrateWithRateFactor = true,
        NeedsEvenSize = false,
    };

    public static readonly FfmpegCodec ProRes = new()
    {
        Id = VideoCodecs.ProRes,
        DisplayName = "ProRes",
        Description = "Apple ProRes: intra-frame, for editing; the preset is the ProRes profile",
        EncoderName = "prores_ks",
        PixelFormats = ["yuv422p10le", "yuv444p10le"],
        Presets = [("proxy", Options(("profile", "0"))), ("lt", Options(("profile", "1"))), ("standard", Options(("profile", "2"))), ("hq", Options(("profile", "3")))],
        DefaultPreset = "hq",
        Quality = QualityControl.None,
    };

    public static readonly FfmpegCodec Ffv1 = new()
    {
        Id = VideoCodecs.Ffv1,
        DisplayName = "FFV1 (lossless)",
        Description = "FFV1: lossless archival video",
        EncoderName = "ffv1",
        IsLossless = true,
        PixelFormats = ["bgr0", "yuv444p", "yuv420p"],
        Quality = QualityControl.None,
        FixedOptions = Options(("level", "3"), ("slicecrc", "1")),
        NeedsEvenSize = false,
    };

    public static readonly FfmpegCodec Apng = new()
    {
        Id = VideoCodecs.Apng,
        DisplayName = "APNG",
        Description = "Animated PNG: lossless, plays in browsers",
        EncoderName = "apng",
        IsLossless = true,
        PixelFormats = ["rgb24"],
        Quality = QualityControl.CompressionLevel,
        FixedOptions = Options(("pred", "mixed")),
        NeedsEvenSize = false,
    };

    public static readonly FfmpegCodec Gif = new()
    {
        Id = VideoCodecs.Gif,
        DisplayName = "GIF",
        Description = "Animated GIF with a palette optimized per clip",
        EncoderName = "gif",
        PixelFormats = ["pal8"],
        Quality = QualityControl.Dither,
        NeedsEvenSize = false,
        UsesPaletteFilter = true,
    };

    public static readonly FfmpegCodec H264Nvenc = Nvenc(VideoCodecs.H264Nvenc, "H.264 (NVIDIA)", "h264_nvenc", ["yuv420p"]);

    public static readonly FfmpegCodec HevcNvenc = Nvenc(VideoCodecs.HevcNvenc, "HEVC (NVIDIA)", "hevc_nvenc", ["yuv420p", "p010le"]);

    public static readonly FfmpegCodec Av1Nvenc = Nvenc(VideoCodecs.Av1Nvenc, "AV1 (NVIDIA)", "av1_nvenc", ["yuv420p", "p010le"]);

    public static readonly FfmpegCodec H264VideoToolbox = VideoToolbox(VideoCodecs.H264VideoToolbox, "H.264 (VideoToolbox)", "h264_videotoolbox");

    public static readonly FfmpegCodec HevcVideoToolbox = VideoToolbox(VideoCodecs.HevcVideoToolbox, "HEVC (VideoToolbox)", "hevc_videotoolbox");

    public static readonly FfmpegCodec H264MediaFoundation = MediaFoundation(VideoCodecs.H264MediaFoundation, "H.264 (Media Foundation)", "h264_mf");

    public static readonly FfmpegCodec HevcMediaFoundation = MediaFoundation(VideoCodecs.HevcMediaFoundation, "HEVC (Media Foundation)", "hevc_mf");

    /// <summary>The hardware encoders, in the order they are offered after the software codecs.</summary>
    public static readonly FfmpegCodec[] Hardware = [H264Nvenc, HevcNvenc, Av1Nvenc, H264VideoToolbox, HevcVideoToolbox, H264MediaFoundation, HevcMediaFoundation];

    private static FfmpegCodec Nvenc(string id, string name, string encoder, string[] pixelFormats) => new()
    {
        Id = id,
        DisplayName = name,
        Description = $"{name}: hardware encoding on the GPU",
        EncoderName = encoder,
        IsHardware = true,
        PixelFormats = pixelFormats,
        Presets = Presets(("preset", ["p1", "p2", "p4", "p6", "p7"])),
        DefaultPreset = VideoPresets.Medium,
        Quality = QualityControl.NvencConstantQuality,
        WorstRateFactor = 40,
        BestRateFactor = 16,
        RateFactorRange = (0, 51),
        FixedOptions = Options(("rc", "vbr"), ("b", "0"), ("tune", "hq")),
    };

    private static FfmpegCodec VideoToolbox(string id, string name, string encoder) => new()
    {
        Id = id,
        DisplayName = name,
        Description = $"{name}: hardware encoding on the Apple media engine",
        EncoderName = encoder,
        IsHardware = true,
        PixelFormats = ["yuv420p", "p010le"],
        Quality = QualityControl.Percent,
        FixedOptions = Options(("allow_sw", "1")),
    };

    private static FfmpegCodec MediaFoundation(string id, string name, string encoder) => new()
    {
        Id = id,
        DisplayName = name,
        Description = $"{name}: the Windows encoder (hardware where available)",
        EncoderName = encoder,
        IsHardware = true,
        PixelFormats = ["nv12"],
        Quality = QualityControl.Percent,
        FixedOptions = Options(("rate_control", "quality"), ("scenario", "archive")),
    };

    /// <summary>libvpx: real-time mode for the fastest preset, "good" with decreasing cpu-used for the others.</summary>
    private static (string, IReadOnlyDictionary<string, string>)[] VpxPresets(int fastestCpuUsed) =>
    [
        (VideoPresets.Fastest, Options(("deadline", "realtime"), ("cpu-used", fastestCpuUsed.ToString(System.Globalization.CultureInfo.InvariantCulture)))),
        (VideoPresets.Fast, Options(("deadline", "good"), ("cpu-used", "4"))),
        (VideoPresets.Medium, Options(("deadline", "good"), ("cpu-used", "2"))),
        (VideoPresets.Slow, Options(("deadline", "good"), ("cpu-used", "1"))),
        (VideoPresets.Slowest, Options(("deadline", "good"), ("cpu-used", "0"))),
    ];

    /// <summary>The five speed presets mapped to the values of one encoder option.</summary>
    private static (string, IReadOnlyDictionary<string, string>)[] Presets((string Option, string[] Values) setting)
    {
        string[] ids = [VideoPresets.Fastest, VideoPresets.Fast, VideoPresets.Medium, VideoPresets.Slow, VideoPresets.Slowest];
        return [.. ids.Select((id, i) => (id, Options((setting.Option, setting.Values[i]))))];
    }

    private static IReadOnlyDictionary<string, string> Options(params (string Key, string Value)[] options) => options.ToDictionary(o => o.Key, o => o.Value);
}
