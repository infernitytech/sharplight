using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Maintenance;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>One line of a storage location card: what uses the space, how much, and its share of what Studio uses there.</summary>
internal sealed class StorageSliceViewModel
{
    public StorageSliceViewModel(StorageUsageItem item, long used)
    {
        Key = item.Key;
        Title = item.Title;
        Bytes = item.Bytes;
        Fraction = used <= 0 ? 0 : Math.Clamp((double)item.Bytes / used, 0, 1);
    }

    /// <summary>The slice (<see cref="UsageKeys"/>).</summary>
    public string Key { get; }

    /// <summary>What uses the space.</summary>
    public string Title { get; }

    /// <summary>The bytes used.</summary>
    public long Bytes { get; }

    /// <summary>The size, e.g. <c>12.4 GB</c>.</summary>
    public string Size => ByteFormat.Format(Bytes);

    /// <summary>The share of what Studio uses in the location, 0..1.</summary>
    public double Fraction { get; }
}
