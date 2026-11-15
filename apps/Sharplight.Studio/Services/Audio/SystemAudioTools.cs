using System.Diagnostics;

namespace Sharplight.Studio.Services.Audio;

/// <summary>Finds the audio programs of the operating system: Studio needs no native audio library, it hands WAV files to what is installed.</summary>
internal static class SystemAudioTools
{
    /// <summary>The full path of <paramref name="program"/> when it is on the <c>PATH</c>, else null.</summary>
    /// <param name="program">A program name without extension.</param>
    public static string? Find(string program)
    {
        string[] extensions = OperatingSystem.IsWindows() ? [".exe", ".cmd", ".bat"] : [""];
        foreach (var directory in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            foreach (var extension in extensions)
            {
                string candidate = Path.Combine(directory, program + extension);
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }
        }

        return null;
    }

    /// <summary>Starts a program with arguments and no console window.</summary>
    /// <param name="program">The program file.</param>
    /// <param name="arguments">The arguments (not shell-parsed).</param>
    public static Process Start(string program, IEnumerable<string> arguments)
    {
        var info = new ProcessStartInfo(program) { UseShellExecute = false, CreateNoWindow = true, RedirectStandardError = true, RedirectStandardOutput = true };
        foreach (var argument in arguments)
        {
            info.ArgumentList.Add(argument);
        }

        return Process.Start(info) ?? throw new InvalidOperationException($"{program} did not start.");
    }
}
