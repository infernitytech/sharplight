using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>One device's memory: what the resident models hold, what else is used, and the total.</summary>
internal sealed class DeviceBudgetViewModel
{
    public DeviceBudgetViewModel(DeviceUsage usage, IReadOnlyList<ILoadedModel> models, long reserveBytes)
    {
        Usage = usage;
        ModelBytes = models.Sum(model => model.ResidentBytes);
        ReserveBytes = reserveBytes;
        ModelNames = string.Join(", ", models.Where(model => model.ResidentBytes > 0).Select(model => $"{model.Id} {ByteFormat.Format(model.ResidentBytes)}"));
    }

    /// <summary>The device's figures.</summary>
    public DeviceUsage Usage { get; }

    /// <summary>Device name.</summary>
    public string Name => Usage.Name;

    /// <summary>Bytes held by the resident models (KV caches, activations and weights).</summary>
    public long ModelBytes { get; }

    /// <summary>Bytes kept free for other applications.</summary>
    public long ReserveBytes { get; }

    /// <summary>Resident models with their memory.</summary>
    public string ModelNames { get; }

    /// <summary>Fraction used in total.</summary>
    public double UsedFraction => Usage.TotalBytes <= 0 ? 0 : Math.Clamp((double)Usage.UsedBytes / Usage.TotalBytes, 0, 1);

    /// <summary>Fraction held by the models.</summary>
    public double ModelFraction => Usage.TotalBytes <= 0 ? 0 : Math.Clamp((double)ModelBytes / Usage.TotalBytes, 0, 1);

    /// <summary>Free bytes beyond the reserve (what a new model may use).</summary>
    public long AvailableBytes => Math.Max(0, Usage.TotalBytes - Usage.UsedBytes - ReserveBytes);

    /// <summary><c>9.1 GB of 24 GB used · 13.9 GB available</c>.</summary>
    public string Text => $"{ByteFormat.Format(Usage.UsedBytes)} of {ByteFormat.Format(Usage.TotalBytes)} used · {ByteFormat.Format(AvailableBytes)} available"
        + (Usage.IsExact ? "" : " (this app only)");
}
