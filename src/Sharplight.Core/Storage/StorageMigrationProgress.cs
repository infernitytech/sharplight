namespace Sharplight.Storage;

/// <summary>Progress of a <see cref="StorageMigration"/>.</summary>
/// <param name="CurrentFile">The file being transferred (relative to its category folder), or null between files.</param>
/// <param name="BytesDone">Bytes transferred so far.</param>
/// <param name="BytesTotal">Bytes to transfer.</param>
/// <param name="FilesDone">Files transferred so far.</param>
/// <param name="FilesTotal">Files to transfer.</param>
public sealed record StorageMigrationProgress(string? CurrentFile, long BytesDone, long BytesTotal, int FilesDone, int FilesTotal)
{
    /// <summary>The fraction done, 0 to 1.</summary>
    public double Fraction => BytesTotal > 0 ? Math.Clamp((double)BytesDone / BytesTotal, 0, 1) : FilesTotal > 0 ? (double)FilesDone / FilesTotal : 1;
}
