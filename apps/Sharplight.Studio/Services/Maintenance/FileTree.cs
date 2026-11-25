namespace Sharplight.Studio.Services.Maintenance;

/// <summary>Size and deletion of folders, never following links (a link in a folder is neither measured through nor deleted through).</summary>
internal static class FileTree
{
    private static readonly EnumerationOptions s_all = new()
    {
        RecurseSubdirectories = true,
        IgnoreInaccessible = true,
        AttributesToSkip = FileAttributes.ReparsePoint,
    };

    /// <summary>The bytes of every file under <paramref name="directory"/> (0 when it does not exist), skipping the folders in <paramref name="excluded"/>.</summary>
    public static long Size(string directory, IReadOnlyCollection<string>? excluded = null, CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(directory))
        {
            return 0;
        }

        long total = 0;
        foreach (var file in new DirectoryInfo(directory).EnumerateFiles("*", s_all))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (excluded is null || !excluded.Any(folder => IsInside(file.FullName, folder)))
            {
                total += file.Length;
            }
        }

        return total;
    }

    /// <summary>Deletes every file and folder under <paramref name="directory"/> but keeps the folder; files in use are skipped.</summary>
    /// <param name="directory">The folder to empty.</param>
    /// <param name="keep">Folders (inside it) that are left alone.</param>
    /// <param name="cancellationToken">Stops between files.</param>
    /// <returns>The bytes deleted.</returns>
    public static long Empty(string directory, IReadOnlyCollection<string>? keep = null, CancellationToken cancellationToken = default)
    {
        if (!Directory.Exists(directory))
        {
            return 0;
        }

        long freed = 0;
        foreach (var file in new DirectoryInfo(directory).EnumerateFiles("*", s_all).ToList())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (keep is not null && keep.Any(folder => IsInside(file.FullName, folder)))
            {
                continue;
            }

            try
            {
                long length = file.Length;
                file.Delete();
                freed += length;
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException)
            {
                // In use by a running load: left for the next time.
            }
        }

        foreach (var folder in new DirectoryInfo(directory).EnumerateDirectories("*", s_all).OrderByDescending(item => item.FullName.Length).ToList())
        {
            if (keep is not null && keep.Any(kept => IsInside(folder.FullName, kept) || IsInside(kept, folder.FullName)))
            {
                continue;
            }

            try
            {
                if (!folder.EnumerateFileSystemInfos().Any())
                {
                    folder.Delete();
                }
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException)
            {
                // Left in place.
            }
        }

        return freed;
    }

    /// <summary>Whether <paramref name="path"/> is <paramref name="folder"/> or lies under it.</summary>
    public static bool IsInside(string path, string folder)
    {
        var comparison = OperatingSystem.IsWindows() || OperatingSystem.IsMacOS() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        var root = Path.GetFullPath(folder).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var full = Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return full.Equals(root, comparison) || full.StartsWith(root + Path.DirectorySeparatorChar, comparison);
    }
}
