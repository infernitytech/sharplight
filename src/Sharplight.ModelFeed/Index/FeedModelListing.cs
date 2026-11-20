namespace Sharplight.ModelFeed.Index;

/// <summary>
/// The models of a channel with their variants (<c>&lt;channel&gt;/catalog/models.json</c>, docs/model-feed.md "Model groups"): one entry per
/// model group, each listing every variant package with what a compatibility check needs, so a reader can show models and compare variants
/// against its hardware without fetching any manifest. Derived from the catalog pages (same channel, same publication) and chained into the
/// signed index by <see cref="FeedCatalogSummary.ModelsDigest"/>.
/// </summary>
public sealed record FeedModelListing
{
    /// <summary>Schema version (1).</summary>
    public int SchemaVersion { get; init; } = 1;

    /// <summary>Model groups, ordered by id.</summary>
    public IReadOnlyList<FeedModel> Models { get; init; } = [];

    /// <summary>
    /// Groups package summaries into models: packages with the same <see cref="FeedPackageSummary.Group"/> id form one model; a package
    /// without a group is a model of its own (id = package id). Variants are ordered by bits per weight, then size, then id. Unlisted
    /// packages are left out.
    /// </summary>
    /// <param name="packages">The channel's package summaries.</param>
    /// <returns>The listing.</returns>
    public static FeedModelListing From(IEnumerable<FeedPackageSummary> packages)
    {
        ArgumentNullException.ThrowIfNull(packages);
        var models = packages
            .Where(package => package.State?.Listed != false)
            .GroupBy(package => package.Group?.Id ?? package.Id, StringComparer.Ordinal)
            .OrderBy(group => group.Key, StringComparer.Ordinal)
            .Select(group => Model(group.Key, [.. group]))
            .ToList();
        return new FeedModelListing { Models = models };
    }

    private static FeedModel Model(string id, IReadOnlyList<FeedPackageSummary> packages)
    {
        var variants = packages
            .Select(FeedModelVariant.From)
            .OrderBy(variant => variant.BitsPerWeight ?? double.MaxValue)
            .ThenBy(variant => variant.Size)
            .ThenBy(variant => variant.Package, StringComparer.Ordinal)
            .ToList();
        var first = packages.FirstOrDefault(package => package.Group?.Recommended == true) ?? packages[0];
        return new FeedModel
        {
            Id = id,
            Name = first.Group?.Name ?? first.Display?.Name,
            Family = first.Display?.Family,
            Kind = first.Model?.Kind,
            Architecture = first.Model?.Architecture,
            Parameters = first.Model?.Parameters,
            ContextLength = first.Model?.ContextLength,
            Publisher = first.Publisher,
            License = first.License,
            Tags = first.Display?.Tags ?? [],
            Recommended = packages.FirstOrDefault(package => package.Group?.Recommended == true)?.Id,
            Variants = variants,
        };
    }
}
