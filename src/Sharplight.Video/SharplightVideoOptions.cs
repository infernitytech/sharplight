namespace Sharplight.Video;

/// <summary>Settings of the video encoders of <c>Sharplight.Video</c>; bind from configuration or set through <c>AddSharplightVideo(configure)</c>.</summary>
public sealed class SharplightVideoOptions
{
    /// <summary>The codec <see cref="WebMVideoEncoder"/> writes. Defaults to <see cref="WebMCodec.Vp9"/>.</summary>
    public WebMCodec WebMCodec { get; set; } = WebMCodec.Vp9;

    /// <summary>
    /// Path or command name of the ffmpeg executable for MP4 output; <see langword="null"/> searches <c>PATH</c>. MP4 (H.264) is patent
    /// encumbered, so no encoder is bundled: <see cref="FfmpegVideoEncoder.IsAvailable"/> is false without ffmpeg.
    /// </summary>
    public string? FfmpegPath { get; set; }
}
