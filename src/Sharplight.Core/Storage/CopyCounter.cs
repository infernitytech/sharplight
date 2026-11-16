namespace Sharplight.Storage;

/// <summary>Running totals of a <see cref="StorageMigration"/> copy, turned into <see cref="StorageMigrationProgress"/> reports.</summary>
/// <param name="totalBytes">Bytes to transfer.</param>
/// <param name="totalFiles">Files to transfer.</param>
internal sealed class CopyCounter(long totalBytes, int totalFiles)
{
    /// <summary>Bytes transferred.</summary>
    public long Bytes { get; private set; }

    /// <summary>Files transferred.</summary>
    public int Files { get; private set; }

    /// <summary>Counts finished files.</summary>
    public void Add(long bytes, int files)
    {
        Bytes += bytes;
        Files += files;
    }

    /// <summary>The progress now, with <paramref name="partialBytes"/> of <paramref name="currentFile"/> copied.</summary>
    public StorageMigrationProgress Snapshot(string? currentFile, long partialBytes = 0) =>
        new(currentFile, Bytes + partialBytes, totalBytes, Files, totalFiles);
}
