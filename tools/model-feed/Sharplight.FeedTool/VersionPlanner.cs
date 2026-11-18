using Sharplight.ModelFeed;
using Sharplight.ModelFeed.Versioning;

namespace Sharplight.FeedTool;

/// <summary>
/// Chooses the version of a newly built package from the feed (docs/model-feed.md section 5): first build 1.0.0; another pinned
/// upstream revision MAJOR+1; the same pins rebuilt (newer engine or rules) MINOR+1; an identical package file with identical
/// metadata nothing; the same package file with changed metadata PATCH+1 (manifest only, reusing the package file). Versions are counted over the shared version folders, so every channel sees one version sequence.
/// </summary>
internal static class VersionPlanner
{
    /// <summary>
    /// Plans the version of a draft manifest: <see langword="null"/> when the feed already has this package file with the same metadata;
    /// a PATCH version reusing the existing package file when only the manifest metadata differs; otherwise the next MAJOR or MINOR.
    /// </summary>
    /// <param name="feedFolder">The feed folder, or <see langword="null"/> when building without a feed.</param>
    /// <param name="draft">The manifest of the new build (any version); its single package file carries the new SHA-256.</param>
    /// <returns>The plan, or <see langword="null"/> when there is nothing to publish.</returns>
    public static VersionPlan? Plan(string? feedFolder, PackageManifest draft)
    {
        ArgumentNullException.ThrowIfNull(draft);
        var manifest = feedFolder is null ? null : LatestManifest(feedFolder, draft.Id);
        if (manifest is null)
        {
            return new VersionPlan("1.0.0", null);
        }

        var latest = SemanticVersion.Parse(manifest.Version);
        var sha256 = draft.PrimaryFile?.Sha256;
        if (string.Equals(manifest.PrimaryFile?.Sha256, sha256, StringComparison.OrdinalIgnoreCase))
        {
            return SameMetadata(manifest, draft) ? null : new VersionPlan(latest.NextPatch().ToString(), manifest);
        }

        var sources = draft.Provenance?.Sources ?? [];
        return new VersionPlan(Pins(manifest.Provenance?.Sources ?? []) == Pins(sources) ? latest.NextMinor().ToString() : latest.NextMajor().ToString(), null);
    }

    /// <summary>
    /// Whether two manifests describe the same package apart from what a rebuild changes by itself: version, file name, build time and
    /// converter, and the minimum engine (taken from the building engine).
    /// </summary>
    /// <param name="published">The published manifest.</param>
    /// <param name="draft">The manifest of the new build.</param>
    /// <returns><see langword="true"/> when the metadata is identical.</returns>
    public static bool SameMetadata(PackageManifest published, PackageManifest draft)
    {
        ArgumentNullException.ThrowIfNull(published);
        ArgumentNullException.ThrowIfNull(draft);
        return Comparable(published, draft) == Comparable(draft, draft);
    }

    // Serializes the first manifest with the rebuild-dependent fields taken from the reference, so only real differences remain.
    private static string Comparable(PackageManifest manifest, PackageManifest reference) =>
        System.Text.Encoding.UTF8.GetString(FeedJson.Write(manifest with
        {
            Version = reference.Version,
            Files = reference.Files,
            Provenance = manifest.Provenance is null ? null : manifest.Provenance with { Built = reference.Provenance?.Built, Converter = reference.Provenance?.Converter },
            Requirements = manifest.Requirements is null ? null : manifest.Requirements with { Engine = reference.Requirements?.Engine },
        }, FeedJsonContext.Default.PackageManifest));

    /// <summary>The manifest of the highest version stored in the feed (any channel), or <see langword="null"/>.</summary>
    /// <param name="feedFolder">The feed folder.</param>
    /// <param name="id">Package id.</param>
    /// <returns>The manifest.</returns>
    public static PackageManifest? LatestManifest(string feedFolder, string id)
    {
        var folder = Path.Combine(feedFolder, "packages", id);
        if (!Directory.Exists(folder))
        {
            return null;
        }

        var latest = Directory.EnumerateDirectories(folder)
            .Where(version => File.Exists(Path.Combine(version, FeedLayout.ManifestFileName)))
            .Select(version => SemanticVersion.TryParse(Path.GetFileName(version), out var parsed) ? parsed : null)
            .Where(version => version is not null)
            .Max();
        return latest is null ? null : FeedJson.ReadManifest(File.ReadAllBytes(Path.Combine(feedFolder, FeedLayout.Manifest(id, latest.ToString()))));
    }

    private static string Pins(IReadOnlyList<ProvenanceSource> sources) =>
        string.Join(";", sources.Select(source => $"{source.Repository}@{source.Revision}").Order(StringComparer.Ordinal));
}
