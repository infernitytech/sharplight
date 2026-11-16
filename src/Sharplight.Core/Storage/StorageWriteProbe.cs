namespace Sharplight.Storage;

/// <summary>Checks that a directory can hold files before a <see cref="StorageLayout"/> uses it.</summary>
public static class StorageWriteProbe
{
    /// <summary>True when <paramref name="directory"/> can be created and a file can be written into it. Creates the directory.</summary>
    /// <param name="directory">The directory.</param>
    public static bool CanWrite(string directory)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        try
        {
            Directory.CreateDirectory(directory);
            var probe = Path.Combine(directory, $".write-probe-{Environment.ProcessId}-{Guid.NewGuid():N}");
            using (new FileStream(probe, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1, FileOptions.DeleteOnClose))
            {
            }

            return true;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }
}
