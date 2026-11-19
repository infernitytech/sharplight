using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Maintenance;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A storage location card: the folder, its volume's total and free space, and what Studio uses in it.</summary>
internal sealed class StorageLocationViewModel
{
    public StorageLocationViewModel(StorageLocationUsage usage)
    {
        Title = usage.Title;
        Path = usage.Path;
        UsedBytes = usage.UsedBytes;
        TotalBytes = usage.TotalBytes;
        FreeBytes = usage.FreeBytes;
        Slices = [.. usage.Items.Select(item => new StorageSliceViewModel(item, usage.UsedBytes))];
    }

    /// <summary>The role of the location (<c>Model store</c>).</summary>
    public string Title { get; }

    /// <summary>The folder.</summary>
    public string Path { get; }

    /// <summary>What Studio uses in the folder.</summary>
    public long UsedBytes { get; }

    /// <summary>The volume's size, or 0 when unknown.</summary>
    public long TotalBytes { get; }

    /// <summary>The volume's free space, or 0 when unknown.</summary>
    public long FreeBytes { get; }

    /// <summary>The slices.</summary>
    public IReadOnlyList<StorageSliceViewModel> Slices { get; }

    /// <summary><c>41.2 GB used by Studio</c>.</summary>
    public string UsedText => $"{ByteFormat.Format(UsedBytes)} used by Studio";

    /// <summary><c>120 GB free of 500 GB</c>, or a note when the volume is unknown.</summary>
    public string VolumeText => TotalBytes > 0 ? $"{ByteFormat.Format(FreeBytes)} free of {ByteFormat.Format(TotalBytes)}" : "Volume size unknown";

    /// <summary>How full the volume is, 0..1 (0 when unknown).</summary>
    public double VolumeFullFraction => TotalBytes > 0 ? Math.Clamp(1 - (double)FreeBytes / TotalBytes, 0, 1) : 0;

    /// <summary>The volume has less than 10 % free (the card warns).</summary>
    public bool IsLow => TotalBytes > 0 && (double)FreeBytes / TotalBytes < 0.10;
}
