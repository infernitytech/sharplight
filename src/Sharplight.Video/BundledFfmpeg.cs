using FFmpeg.AutoGen;
using Sharplight.Video.Ffmpeg;

namespace Sharplight.Video;

/// <summary>
/// The FFmpeg libraries Sharplight ships (its own LGPL build, native/ffmpeg): whether they loaded and what they are. Loading happens once
/// per process on first use; it is the same for every encoder and decoder instance.
/// </summary>
public static class BundledFfmpeg
{
    /// <summary>Environment variable naming a directory with the FFmpeg libraries; searched before the application's own.</summary>
    public const string DirectoryVariable = "SHARPLIGHT_FFMPEG_DIR";

    /// <summary>Whether the libraries for this platform were found and match the bindings' major versions.</summary>
    public static bool IsAvailable => FfmpegLoader.Result.Error is null;

    /// <summary>Why <see cref="IsAvailable"/> is false; <see langword="null"/> when the libraries loaded.</summary>
    public static string? UnavailableReason => FfmpegLoader.Result.Error;

    /// <summary>The FFmpeg version, for example <c>9.0.2</c>; empty when unavailable.</summary>
    public static string Version => IsAvailable ? ffmpeg.av_version_info() ?? string.Empty : string.Empty;

    /// <summary>The license FFmpeg was configured with (an LGPL text for Sharplight's build); empty when unavailable.</summary>
    public static string License => IsAvailable ? ffmpeg.avutil_license() ?? string.Empty : string.Empty;

    /// <summary>The file names of the libraries on this platform, in load order (what a published application must contain).</summary>
    public static IReadOnlyList<string> LibraryFileNames => [.. FfmpegLoader.Libraries.Select(FfmpegLoader.FileName)];

    /// <summary>Throws <see cref="PlatformNotSupportedException"/> when the libraries are unavailable.</summary>
    internal static void ThrowIfUnavailable()
    {
        string? error = FfmpegLoader.Result.Error;
        if (error is not null)
        {
            throw new PlatformNotSupportedException(error);
        }
    }
}
