using System.Text.Json.Serialization;

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

    /// <summary>
    /// Where the files are downloaded from (<see cref="CatalogSources"/>): <see langword="null"/> for the Hugging Face Hub, <c>github</c>
    /// for the files of a GitHub repository at <see cref="Revision"/>, <c>github-release</c> for the assets of the GitHub release
    /// <see cref="Release"/>. Files from GitHub carry a sha256 each, checked after the download like the Hub's LFS files.
    /// </summary>
    public string? Source { get; init; }

    /// <summary>
    /// The GitHub release tag the files belong to: the release whose assets a <c>github-release</c> set downloads, or the release a
    /// <c>github</c> set's <see cref="Revision"/> was resolved from (the tag's commit); <see langword="null"/> otherwise.
    /// </summary>
    public string? Release { get; init; }

    /// <summary>Whether the files come from the Hugging Face Hub (no <see cref="Source"/>, or <c>huggingface</c>).</summary>
    [JsonIgnore]
    public bool IsHubSource => Source is null or CatalogSources.HuggingFace;

    /// <summary>
    /// The download address of one of <see cref="Files"/> when the set is not on the Hugging Face Hub, pinned to <see cref="Revision"/>
    /// (repository files) or to the release's asset; <see langword="null"/> for Hub sets, whose addresses the <see cref="HubClient"/> makes.
    /// </summary>
    /// <param name="path">The file's path in the set.</param>
    public Uri? GetDirectFileUri(string path) => Source switch
    {
        CatalogSources.GitHub => new Uri(CatalogSources.GitHubRaw, $"{Repository}/{Revision}/{CatalogSources.EscapePath(path)}"),
        CatalogSources.GitHubRelease => new Uri(CatalogSources.GitHubWeb, $"{Repository}/releases/download/{Uri.EscapeDataString(Release ?? "")}/{CatalogSources.EscapePath(path)}"),
        _ => null,
    };

    /// <summary>The web page of the pinned files: the repository tree at <see cref="Revision"/>, or the GitHub release page.</summary>
    /// <param name="hubEndpoint">The Hugging Face Hub address (<see cref="HubClientOptions.Endpoint"/>), for Hub sets.</param>
    public Uri GetPageUri(Uri hubEndpoint)
    {
        ArgumentNullException.ThrowIfNull(hubEndpoint);
        return Source switch
        {
            CatalogSources.GitHub => new Uri(CatalogSources.GitHubWeb, $"{Repository}/tree/{Revision}"),
            CatalogSources.GitHubRelease => new Uri(CatalogSources.GitHubWeb, $"{Repository}/releases/tag/{Uri.EscapeDataString(Release ?? "")}"),
            _ => new Uri(hubEndpoint, $"{Repository}/tree/{Revision}"),
        };
    }
}
