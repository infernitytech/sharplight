namespace Sharplight.Video.Tests;

/// <summary>Probes and decodes files with the bundled FFmpeg decoder, as the reference for the encoder tests.</summary>
internal static class FfmpegReference
{
    /// <summary>Whether the bundled FFmpeg libraries loaded.</summary>
    public static bool IsInstalled => BundledFfmpeg.IsAvailable;

    /// <summary>Codec name, size and decoded frame count of the first video stream.</summary>
    public static (string Codec, int Width, int Height, int Frames) Probe(string path)
    {
        var decoder = new FfmpegVideoDecoder();
        VideoInfo info = decoder.ProbeAsync(path).GetAwaiter().GetResult();
        return (info.VideoCodec, info.Width, info.Height, Decode(path).Count);
    }

    /// <summary>Decodes the video to RGB frames at its own size.</summary>
    public static List<byte[]> Decode(string path)
    {
        return DecodeAsync(path).GetAwaiter().GetResult();
    }

    private static async Task<List<byte[]>> DecodeAsync(string path)
    {
        var frames = new List<byte[]>();
        await foreach (VideoFrame frame in new FfmpegVideoDecoder().ReadFramesAsync(path))
        {
            frames.Add(frame.Data);
        }

        return frames;
    }
}
