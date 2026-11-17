namespace Sharplight.ModelFeed.Publishing;

/// <summary>What a publication wrote.</summary>
/// <param name="Sequence">The new index sequence.</param>
/// <param name="Added">Package versions added (<c>id@version</c>).</param>
/// <param name="Unchanged">Package versions already present with identical manifests.</param>
/// <param name="Packages">Packages in the catalog.</param>
public sealed record FeedPublishResult(long Sequence, IReadOnlyList<string> Added, IReadOnlyList<string> Unchanged, int Packages);
