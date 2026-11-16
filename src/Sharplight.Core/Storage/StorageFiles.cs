namespace Sharplight.Storage;

/// <summary>Small file helpers shared by the storage types.</summary>
internal static class StorageFiles
{
    private const string TemporarySuffix = ".tmp";

    /// <summary>Writes <paramref name="content"/> to a temporary file that then replaces <paramref name="path"/>, creating the folder.</summary>
    public static void WriteAtomically(string path, string content)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        var temporary = path + TemporarySuffix;
        File.WriteAllText(temporary, content);
        File.Move(temporary, path, overwrite: true);
    }

    /// <summary>Deletes <paramref name="path"/> when it exists.</summary>
    public static void DeleteIfExists(string path)
    {
        if (File.Exists(path))
        {
            File.Delete(path);
        }
    }
}
