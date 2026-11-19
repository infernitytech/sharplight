namespace Sharplight.Studio.Services.Maintenance;

/// <summary>What Studio uses in every storage location, and the partial downloads that could be removed.</summary>
/// <param name="Locations">The model store, data, cache and logs locations.</param>
/// <param name="PartialDownloads">Leftovers of failed or cancelled downloads in the model store.</param>
internal sealed record StorageOverview(IReadOnlyList<StorageLocationUsage> Locations, IReadOnlyList<PartialDownload> PartialDownloads)
{
    /// <summary>Bytes of <see cref="PartialDownloads"/>.</summary>
    public long PartialBytes => PartialDownloads.Sum(item => item.Bytes);
}
