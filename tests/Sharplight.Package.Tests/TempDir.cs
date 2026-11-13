namespace Sharplight.Package.Tests;

/// <summary>A uniquely named temporary directory that is deleted (best effort) on disposal.</summary>
public sealed class TempDir : IDisposable
{
    /// <summary>Absolute path of the directory.</summary>
    public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "sharplight-pkg-" + Guid.NewGuid().ToString("N")[..8]);

    /// <summary>Creates the directory.</summary>
    public TempDir() => Directory.CreateDirectory(Path);

    /// <summary>Returns the path of a file with the given name inside the directory.</summary>
    public string File(string name) => System.IO.Path.Combine(Path, name);

    /// <inheritdoc/>
    public void Dispose()
    {
        try
        {
            Directory.Delete(Path, true);
        }
        catch (Exception)
        {
            // Leftover temp files are harmless.
        }
    }
}
