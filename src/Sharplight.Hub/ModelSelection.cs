namespace Sharplight.Hub;

/// <summary>The files of one model variant to download, as chosen by <see cref="ModelFileSelector"/>.</summary>
/// <param name="Format">The variant's format.</param>
/// <param name="Tag">The variant's tag (e.g. <c>Q4_K_M</c>); empty when the repository has a single variant.</param>
/// <param name="Files">Every file to download, including auxiliary ones (tokenizer, index, vision projector, external data).</param>
/// <param name="PrimaryPath">The file or directory (repository path; empty for the root) a converter opens.</param>
/// <param name="ProjectorPath">The vision projector to build into the package, or <see langword="null"/>.</param>
public sealed record ModelSelection(
    ModelFormat Format,
    string Tag,
    IReadOnlyList<HubFile> Files,
    string PrimaryPath,
    string? ProjectorPath)
{
    /// <summary>Total size of <see cref="Files"/> in bytes.</summary>
    public long TotalBytes => Files.Sum(file => file.Size);
}
