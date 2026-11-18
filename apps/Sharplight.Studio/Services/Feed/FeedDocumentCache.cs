namespace Sharplight.Studio.Services.Feed;

/// <summary>The last verified documents of one feed channel on disk: what the offline listing is built from.</summary>
internal sealed class FeedDocumentCache(string directory)
{
    /// <summary>Stores a document (atomically).</summary>
    public void Write(string path, byte[] bytes)
    {
        var file = FileFor(path);
        Directory.CreateDirectory(Path.GetDirectoryName(file)!);
        var temporary = file + ".tmp";
        File.WriteAllBytes(temporary, bytes);
        File.Move(temporary, file, overwrite: true);
    }

    /// <summary>The stored document, or null.</summary>
    public byte[]? Read(string path)
    {
        var file = FileFor(path);
        return File.Exists(file) ? File.ReadAllBytes(file) : null;
    }

    private string FileFor(string path) => Path.Combine(directory, path.Replace('/', Path.DirectorySeparatorChar));
}
