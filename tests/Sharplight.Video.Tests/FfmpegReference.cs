using System.Diagnostics;
using System.Globalization;

namespace Sharplight.Video.Tests;

/// <summary>Decodes and probes files with the ffmpeg command line tools as an independent reference, when they are installed.</summary>
internal static class FfmpegReference
{
    private static readonly Lazy<bool> s_installed = new(() => Run("ffmpeg", ["-version"]) is not null && Run("ffprobe", ["-version"]) is not null);
    private static readonly Lazy<bool> s_h264 = new(() => Run("ffmpeg", ["-hide_banner", "-encoders"])?.Contains("libx264", StringComparison.Ordinal) == true);

    /// <summary>Whether ffmpeg and ffprobe run.</summary>
    public static bool IsInstalled => s_installed.Value;

    /// <summary>Whether this ffmpeg can encode H.264 with libx264.</summary>
    public static bool HasH264 => IsInstalled && s_h264.Value;

    /// <summary>Codec name, size and decoded frame count of the first video stream.</summary>
    public static (string Codec, int Width, int Height, int Frames) Probe(string path)
    {
        string text = Run("ffprobe", ["-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=codec_name,width,height,nb_read_frames", "-of", "csv=p=0", path])
            ?? throw new InvalidOperationException("ffprobe failed on " + path);
        string[] parts = text.Trim().Split(',');
        return (parts[0], int.Parse(parts[1], CultureInfo.InvariantCulture), int.Parse(parts[2], CultureInfo.InvariantCulture), int.Parse(parts[3], CultureInfo.InvariantCulture));
    }

    /// <summary>Decodes the video to RGB frames.</summary>
    public static List<byte[]> Decode(string path, int width, int height)
    {
        string output = TestClips.TempPath(".rgb");
        try
        {
            Run("ffmpeg", ["-v", "error", "-y", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", output]);
            byte[] all = File.ReadAllBytes(output);
            int size = width * height * 3;
            var frames = new List<byte[]>();
            for (int offset = 0; offset + size <= all.Length; offset += size)
            {
                frames.Add(all[offset..(offset + size)]);
            }

            return frames;
        }
        finally
        {
            File.Delete(output);
        }
    }

    private static string? Run(string program, string[] arguments)
    {
        try
        {
            var info = new ProcessStartInfo(program)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            foreach (string argument in arguments)
            {
                info.ArgumentList.Add(argument);
            }

            using Process? process = Process.Start(info);
            if (process is null)
            {
                return null;
            }

            string standardOutput = process.StandardOutput.ReadToEnd();
            process.StandardError.ReadToEnd();
            process.WaitForExit();
            return process.ExitCode == 0 ? standardOutput : null;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return null;
        }
    }
}
