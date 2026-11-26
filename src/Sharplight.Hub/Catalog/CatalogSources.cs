namespace Sharplight.Hub.Catalog;

/// <summary>
/// The <see cref="CatalogFileSet.Source"/> values: where a file set is downloaded from. Most publishers release on the Hugging Face Hub;
/// some only on GitHub (Silero VAD), whose files are downloaded from the repository at the pinned commit or from the assets of a release.
/// </summary>
public static class CatalogSources
{
    /// <summary>A Hugging Face Hub model repository (the default: the catalog omits the field).</summary>
    public const string HuggingFace = "huggingface";

    /// <summary>Files of a GitHub repository at the pinned commit (<c>raw.githubusercontent.com</c>).</summary>
    public const string GitHub = "github";

    /// <summary>Assets of a GitHub release (<see cref="CatalogFileSet.Release"/>, a tag that points at the pinned commit).</summary>
    public const string GitHubRelease = "github-release";

    /// <summary>Every known source.</summary>
    public static IReadOnlyList<string> All { get; } = [HuggingFace, GitHub, GitHubRelease];

    /// <summary>The GitHub web site: repository pages and release asset downloads.</summary>
    public static Uri GitHubWeb { get; } = new("https://github.com/");

    /// <summary>The host that serves GitHub repository files at a commit.</summary>
    public static Uri GitHubRaw { get; } = new("https://raw.githubusercontent.com/");

    /// <summary>The GitHub REST API.</summary>
    public static Uri GitHubApi { get; } = new("https://api.github.com/");

    /// <summary>Whether <paramref name="source"/> names a GitHub source.</summary>
    /// <param name="source">A <see cref="CatalogFileSet.Source"/> value.</param>
    public static bool IsGitHub(string? source) => source is GitHub or GitHubRelease;

    /// <summary>A repository path with every segment escaped for a URL.</summary>
    /// <param name="path">A path with <c>/</c> separators.</param>
    internal static string EscapePath(string path) => string.Join('/', path.Split('/').Select(Uri.EscapeDataString));
}
