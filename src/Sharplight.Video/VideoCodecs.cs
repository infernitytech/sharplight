namespace Sharplight.Video;

/// <summary>Ids of the video codecs of the bundled FFmpeg encoders (<see cref="VideoEncodeOptions.Codec"/>).</summary>
public static class VideoCodecs
{
    /// <summary>H.264 / AVC through OpenH264 (MP4, MOV, Matroska); plays everywhere.</summary>
    public const string H264 = "h264";

    /// <summary>AV1 through SVT-AV1 (MP4, WebM, Matroska); the smallest files, slower to encode.</summary>
    public const string Av1 = "av1";

    /// <summary>VP9 through libvpx (WebM, MP4, Matroska).</summary>
    public const string Vp9 = "vp9";

    /// <summary>VP8 through libvpx (WebM).</summary>
    public const string Vp8 = "vp8";

    /// <summary>Apple ProRes 422 / 4444 (MOV, Matroska), an intra-frame editing codec.</summary>
    public const string ProRes = "prores";

    /// <summary>FFV1, lossless (Matroska, MOV).</summary>
    public const string Ffv1 = "ffv1";

    /// <summary>Animated PNG (lossless).</summary>
    public const string Apng = "apng";

    /// <summary>GIF with an optimized palette.</summary>
    public const string Gif = "gif";

    /// <summary>H.264 on an NVIDIA GPU (NVENC).</summary>
    public const string H264Nvenc = "h264-nvenc";

    /// <summary>H.265 / HEVC on an NVIDIA GPU (NVENC).</summary>
    public const string HevcNvenc = "hevc-nvenc";

    /// <summary>AV1 on an NVIDIA GPU (NVENC, RTX 40 series and newer).</summary>
    public const string Av1Nvenc = "av1-nvenc";

    /// <summary>H.264 through Apple VideoToolbox (macOS).</summary>
    public const string H264VideoToolbox = "h264-videotoolbox";

    /// <summary>H.265 / HEVC through Apple VideoToolbox (macOS).</summary>
    public const string HevcVideoToolbox = "hevc-videotoolbox";

    /// <summary>H.264 through Windows Media Foundation.</summary>
    public const string H264MediaFoundation = "h264-mf";

    /// <summary>H.265 / HEVC through Windows Media Foundation (needs the HEVC extension or a hardware encoder).</summary>
    public const string HevcMediaFoundation = "hevc-mf";

    /// <summary>AAC audio (FFmpeg's native encoder).</summary>
    public const string Aac = "aac";

    /// <summary>Opus audio (libopus).</summary>
    public const string Opus = "opus";

    /// <summary>FLAC audio, lossless.</summary>
    public const string Flac = "flac";

    /// <summary>Uncompressed 16-bit PCM audio.</summary>
    public const string Pcm = "pcm";
}
