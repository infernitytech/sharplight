namespace Sharplight.Studio.Services.Maintenance;

/// <summary>The usage of one storage location (model store, data, cache or logs): the volume's figures and what Studio uses in it.</summary>
/// <param name="Title">The location's role, e.g. <c>Model store</c>.</param>
/// <param name="Path">The folder.</param>
/// <param name="TotalBytes">Size of the volume it is on; 0 when unknown.</param>
/// <param name="FreeBytes">Free space of the volume; 0 when unknown.</param>
/// <param name="Items">What Studio uses in the folder.</param>
internal sealed record StorageLocationUsage(string Title, string Path, long TotalBytes, long FreeBytes, IReadOnlyList<StorageUsageItem> Items)
{
    /// <summary>The sum of <see cref="Items"/>.</summary>
    public long UsedBytes => Items.Sum(item => item.Bytes);
}
