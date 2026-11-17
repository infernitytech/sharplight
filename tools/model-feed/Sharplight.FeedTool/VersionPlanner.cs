using Sharplight.ModelFeed;
using Sharplight.ModelFeed.Versioning;

namespace Sharplight.FeedTool;

/// <summary>
/// Chooses the version of a newly built package from the feed (docs/model-feed.md section 5): first build 1.0.0; another pinned
/// upstream revision MAJOR+1; the same pins rebuilt (newer engine or rules) MINOR+1; an identical package file nothing.
/// </summary>
internal static class VersionPlanner
{
    /// <summary>The next version, or <see langword="null"/> when the feed already has an identical package file.</summary>
    /// <param name="feedFolder">The feed folder, or <see langword="null"/> when building without a feed.</param>
    /// <param name="id">Package id.</param>
    /// <param name="sources">The upstream sources of the new build.</param>
    /// <param name="sha256">SHA-256 of the new package file.</param>
    /// <returns>The version.</returns>
    public static string? Next(string? feedFolder, string id, IReadOnlyList<ProvenanceSource> sources, string sha256)
    {
        if (feedFolder is null)
        {
            return "1.0.0";
        }

        var indexPath = Path.Combine(feedFolder, FeedLayout.PackageIndex(id));
        if (!File.Exists(indexPath))
        {
            return "1.0.0";
        }

        var index = FeedJson.Read(File.ReadAllBytes(indexPath), FeedJsonContext.Default.FeedPackageIndex, $"The package index of {id}");
        var latest = index.Versions.Select(entry => SemanticVersion.Parse(entry.Version)).Max();
        if (latest is null)
        {
            return "1.0.0";
        }

        var manifest = FeedJson.ReadManifest(File.ReadAllBytes(Path.Combine(feedFolder, FeedLayout.Manifest(id, latest.ToString()))));
        if (string.Equals(manifest.PrimaryFile?.Sha256, sha256, StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        return Pins(manifest.Provenance?.Sources ?? []) == Pins(sources) ? latest.NextMinor().ToString() : latest.NextMajor().ToString();
    }

    private static string Pins(IReadOnlyList<ProvenanceSource> sources) =>
        string.Join(";", sources.Select(source => $"{source.Repository}@{source.Revision}").Order(StringComparer.Ordinal));
}
