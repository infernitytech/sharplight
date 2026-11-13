namespace Sharplight.Hub;

/// <summary>A file of a repository at a pinned commit, as listed by the Hub's tree API.</summary>
/// <param name="Path">Path inside the repository, with <c>/</c> separators.</param>
/// <param name="Size">Size in bytes (the real size for LFS files, not the pointer's).</param>
/// <param name="Sha256">Lower-case hex SHA-256 of the content for LFS files, otherwise <see langword="null"/>.</param>
/// <param name="GitOid">Lower-case hex git blob id (SHA-1 over <c>"blob {size}\0" + content</c>) for files stored in git, otherwise <see langword="null"/>.</param>
public sealed record HubFile(string Path, long Size, string? Sha256, string? GitOid)
{
    /// <summary>The file name without directories.</summary>
    public string Name => RepoPath.Name(Path);

    /// <summary>The directory part of <see cref="Path"/> (empty for files at the repository root).</summary>
    public string Directory => RepoPath.Directory(Path);
}
