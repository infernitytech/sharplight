namespace Sharplight.Hub;

/// <summary>Splits repository paths (always <c>/</c>-separated, independent of the local platform).</summary>
internal static class RepoPath
{
    private const char Separator = '/';

    /// <summary>The file name without directories.</summary>
    public static string Name(string path) => path[(path.LastIndexOf(Separator) + 1)..];

    /// <summary>The directory part (empty at the repository root).</summary>
    public static string Directory(string path) => path.LastIndexOf(Separator) is var slash and >= 0 ? path[..slash] : "";

    /// <summary>True for files at the repository root.</summary>
    public static bool IsRoot(string path) => !path.Contains(Separator);

    /// <summary>The local path of <paramref name="path"/> under <paramref name="root"/>.</summary>
    /// <exception cref="HubException">The path would leave <paramref name="root"/> (<c>..</c> segments or an absolute path).</exception>
    public static string ToLocal(string root, string path)
    {
        var segments = path.Split(Separator);
        if (path.Length == 0 || segments.Any(segment => segment is "" or "." or ".."))
        {
            throw new HubException($"Refusing to store the repository path '{path}'.");
        }

        return System.IO.Path.Combine([root, .. segments]);
    }
}
