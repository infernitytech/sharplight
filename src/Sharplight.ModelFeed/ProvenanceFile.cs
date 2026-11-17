namespace Sharplight.ModelFeed;

/// <summary>A source file with its checksum.</summary>
public sealed record ProvenanceFile
{
    /// <summary>Path in the repository.</summary>
    public required string Path { get; init; }

    /// <summary>Size in bytes.</summary>
    public long Size { get; init; }

    /// <summary>SHA-256 (large files).</summary>
    public string? Sha256 { get; init; }

    /// <summary>Git blob id (small files).</summary>
    public string? GitOid { get; init; }
}
