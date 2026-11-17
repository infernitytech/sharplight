namespace Sharplight.Video;

/// <summary>Settings of the video services of <c>Sharplight.Video</c>; bind from configuration or set through <c>AddSharplightVideo(configure)</c>.</summary>
public sealed class SharplightVideoOptions
{
    /// <summary>
    /// Whether the encoders offer the hardware codecs (NVENC, VideoToolbox, Media Foundation) that work on this machine. Each is tried once
    /// per process with a small trial encode. Defaults to true; ignored on Windows, where only the software codecs are offered.
    /// </summary>
    public bool EnableHardwareEncoders { get; set; } = true;
}
