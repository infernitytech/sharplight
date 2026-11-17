namespace Sharplight.Hosting.Loaders;

/// <summary>
/// Rough footprints from file sizes, for models whose exact size is only known once loaded: the weights plus a quarter for activations
/// and state plus a fixed margin.
/// </summary>
internal static class FileFootprint
{
    private const double OverheadFactor = 1.25;
    private const long OverheadBytes = 256L << 20;

    /// <summary>The estimated device bytes of the model at <paramref name="path"/> with its auxiliary files.</summary>
    public static long Estimate(string path, IEnumerable<string> auxiliary)
    {
        long bytes = (long)(Bytes(path) * OverheadFactor) + OverheadBytes;
        foreach (var file in auxiliary)
        {
            bytes += Bytes(file);
        }

        return bytes;
    }

    /// <summary>Size of a file, or of every file under a directory; 0 when it does not exist.</summary>
    public static long Bytes(string path)
    {
        if (File.Exists(path))
        {
            return new FileInfo(path).Length;
        }

        return Directory.Exists(path)
            ? new DirectoryInfo(path).EnumerateFiles("*", SearchOption.AllDirectories).Sum(file => file.Length)
            : 0;
    }
}
