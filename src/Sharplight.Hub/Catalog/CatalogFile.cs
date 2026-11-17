namespace Sharplight.Hub.Catalog;

/// <summary>A file of a pinned repository revision.</summary>
public sealed record CatalogFile
{
    /// <summary>Path inside the repository, with <c>/</c> separators.</summary>
    public required string Path { get; init; }

    /// <summary>Size in bytes.</summary>
    public long Size { get; init; }

    /// <summary>Lower-case hex SHA-256 of the content (the Hub's LFS object id), or <see langword="null"/> for files stored in git.</summary>
    public string? Sha256 { get; init; }

    /// <summary>Lower-case hex git blob id for files stored in git, or <see langword="null"/> for LFS files.</summary>
    public string? GitOid { get; init; }
}
