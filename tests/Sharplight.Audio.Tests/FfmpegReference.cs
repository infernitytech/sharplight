using System.Diagnostics;

namespace Sharplight.Audio.Tests;

/// <summary>Decodes a file with the ffmpeg command line tool as a reference, when it is installed.</summary>
internal static class FfmpegReference
{
    public static float[]? Decode(string path)
    {
        try
        {
            string output = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"sl-ref-{Guid.NewGuid():N}.raw");
            var info = new ProcessStartInfo("ffmpeg")
            {
                RedirectStandardError = true,
                RedirectStandardOutput = true,
            };
            foreach (string argument in new[] { "-v", "error", "-y", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le", output })
            {
                info.ArgumentList.Add(argument);
            }

            using Process? process = Process.Start(info);
            if (process is null)
            {
                return null;
            }

            process.WaitForExit();
            if (process.ExitCode != 0)
            {
                return null;
            }

            byte[] bytes = File.ReadAllBytes(output);
            File.Delete(output);
            var samples = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, samples, 0, samples.Length * 4);
            return samples;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return null;
        }
    }
}
