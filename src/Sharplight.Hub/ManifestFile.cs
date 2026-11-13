namespace Sharplight.Hub;

/// <summary>A source file a stored model was converted from.</summary>
public sealed class ManifestFile
{
    /// <summary>Path in the repository.</summary>
    public required string Path { get; init; }

    /// <summary>Size in bytes.</summary>
    public long Size { get; init; }

    /// <summary>SHA-256 of the content (LFS files), or <see langword="null"/>.</summary>
    public string? Sha256 { get; init; }
}
