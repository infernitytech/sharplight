namespace Sharplight.Video;

/// <summary>Identifiers of the video formats (<see cref="IVideoEncoder.Format"/>).</summary>
public static class VideoFormats
{
    /// <summary>WebM (<c>.webm</c>).</summary>
    public const string WebM = "webm";

    /// <summary>MP4 (<c>.mp4</c>).</summary>
    public const string Mp4 = "mp4";

    /// <summary>QuickTime movie (<c>.mov</c>), for editing workflows.</summary>
    public const string Mov = "mov";

    /// <summary>Matroska (<c>.mkv</c>).</summary>
    public const string Matroska = "matroska";

    /// <summary>Animated PNG (<c>.apng</c>, <c>.png</c>), lossless.</summary>
    public const string Apng = "apng";

    /// <summary>Animated GIF (<c>.gif</c>), 256 colors per frame.</summary>
    public const string Gif = "gif";
}
