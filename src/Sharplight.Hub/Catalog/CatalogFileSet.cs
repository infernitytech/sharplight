namespace Sharplight.Hub.Catalog;

/// <summary>
/// Files of one Hub repository pinned to a commit: what a variant or a companion downloads. <see cref="Include"/> and
/// <see cref="Tag"/> are how the set was chosen (the catalog tool resolves them again with <c>resolve --refresh</c>);
/// <see cref="Files"/> is the exact list that is installed.
/// </summary>
public abstract record CatalogFileSet
{
    /// <summary>The repository id, <c>owner/name</c>.</summary>
    public required string Repository { get; init; }

    /// <summary>The commit (40 hex digits) the files were verified at; downloads are pinned to it.</summary>
    public string Revision { get; init; } = "";

    /// <summary>
    /// The variant tag the files were chosen with (<see cref="ModelFileSelector"/>, e.g. <c>Q4_K_M</c>), or <see langword="null"/>
    /// when <see cref="Include"/> names them or the repository has a single variant.
    /// </summary>
    public string? Tag { get; init; }

    /// <summary>Glob patterns (<c>*</c>, <c>**</c>, <c>?</c>) of the repository paths to take; empty to let the selector choose.</summary>
    public IReadOnlyList<string> Include { get; init; } = [];

    /// <summary>The repository path a converter opens: a file, a directory, or empty for the repository root.</summary>
    public string Primary { get; init; } = "";

    /// <summary>Every file to download.</summary>
    public IReadOnlyList<CatalogFile> Files { get; init; } = [];

    /// <summary>Total size of <see cref="Files"/> in bytes.</summary>
    public long TotalBytes => Files.Sum(file => file.Size);
}
