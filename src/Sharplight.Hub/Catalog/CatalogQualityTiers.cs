namespace Sharplight.Hub.Catalog;

/// <summary>
/// Quality tiers of <see cref="CatalogVariant.Quality"/>, best first; the model feed publishes them unchanged as the variant's
/// <c>group.quality</c> (Sharplight.ModelFeed.PackageQualityTiers, docs/model-feed.md "Model groups").
/// </summary>
public static class CatalogQualityTiers
{
    /// <summary>Every tier, best first: reference, near-lossless (8-bit, FP8), high (5-6 bit), good (4 bit), reduced (3 bit), low (1-2 bit).</summary>
    public static IReadOnlyList<string> All { get; } = ["reference", "near-lossless", "high", "good", "reduced", "low"];
}
