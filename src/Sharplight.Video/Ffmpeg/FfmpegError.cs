using FFmpeg.AutoGen;

namespace Sharplight.Video.Ffmpeg;

/// <summary>Turns FFmpeg's negative error codes into exceptions with FFmpeg's message.</summary>
internal static unsafe class FfmpegError
{
    /// <summary>FFmpeg's text for an error code.</summary>
    public static string Message(int error)
    {
        const int size = 256;
        byte* buffer = stackalloc byte[size];
        ffmpeg.av_strerror(error, buffer, size);
        return System.Runtime.InteropServices.Marshal.PtrToStringUTF8((nint)buffer) ?? $"error {error}";
    }

    /// <summary>Throws <see cref="InvalidOperationException"/> when <paramref name="result"/> is negative.</summary>
    /// <returns>The non-negative result.</returns>
    public static int Check(int result, string action)
    {
        if (result < 0)
        {
            throw new InvalidOperationException($"FFmpeg could not {action}: {Message(result)}.");
        }

        return result;
    }

    /// <summary>Throws <see cref="InvalidDataException"/> (unreadable input) when <paramref name="result"/> is negative.</summary>
    /// <returns>The non-negative result.</returns>
    public static int CheckInput(int result, string action)
    {
        if (result < 0)
        {
            throw new InvalidDataException($"FFmpeg could not {action}: {Message(result)}.");
        }

        return result;
    }

    /// <summary>Whether the code means "no output now, send more input" or "end of stream".</summary>
    public static bool IsAgainOrEnd(int result) => result == ffmpeg.AVERROR(ffmpeg.EAGAIN) || result == ffmpeg.AVERROR_EOF;
}
