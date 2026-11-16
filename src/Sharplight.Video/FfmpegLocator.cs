namespace Sharplight.Video;

/// <summary>Finds the ffmpeg executable: an explicit path or command name, otherwise the first match on <c>PATH</c>.</summary>
internal static class FfmpegLocator
{
    /// <summary>Returns the full path of the executable, or <see langword="null"/> when there is none.</summary>
    public static string? Find(string? configured)
    {
        string name = string.IsNullOrWhiteSpace(configured) ? "ffmpeg" : configured;
        if (name.Contains(Path.DirectorySeparatorChar) || name.Contains(Path.AltDirectorySeparatorChar))
        {
            return File.Exists(name) ? Path.GetFullPath(name) : null;
        }

        string[] extensions = OperatingSystem.IsWindows() ? [".exe", ".cmd", ".bat", string.Empty] : [string.Empty];
        string path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        foreach (string directory in path.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            foreach (string extension in extensions)
            {
                string candidate = Path.Combine(directory.Trim('"'), name + extension);
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }
        }

        return null;
    }
}
