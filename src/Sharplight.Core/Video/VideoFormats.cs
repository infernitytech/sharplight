namespace Sharplight.Video;

/// <summary>Identifiers of the video formats (<see cref="IVideoEncoder.Format"/>).</summary>
public static class VideoFormats
{
    /// <summary>WebM with VP8 or VP9 video (<c>.webm</c>).</summary>
    public const string WebM = "webm";

    /// <summary>MP4 with H.264 video (<c>.mp4</c>); needs an external ffmpeg.</summary>
    public const string Mp4 = "mp4";

    /// <summary>Animated PNG (<c>.apng</c>, <c>.png</c>), lossless.</summary>
    public const string Apng = "apng";

    /// <summary>Animated GIF (<c>.gif</c>), 256 colors per frame.</summary>
    public const string Gif = "gif";
}
