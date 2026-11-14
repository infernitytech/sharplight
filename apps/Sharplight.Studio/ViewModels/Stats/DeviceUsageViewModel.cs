using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Stats;

/// <summary>Memory bar of one device.</summary>
internal sealed class DeviceUsageViewModel
{
    public DeviceUsageViewModel(DeviceUsage usage)
    {
        Usage = usage;
    }

    /// <summary>The figures.</summary>
    public DeviceUsage Usage { get; }

    /// <summary>Device name.</summary>
    public string Name => Usage.Name;

    /// <summary>Used fraction.</summary>
    public double Fraction => Usage.TotalBytes <= 0 ? 0 : Math.Clamp((double)Usage.UsedBytes / Usage.TotalBytes, 0, 1);

    /// <summary><c>3.2 GB / 24 GB</c>.</summary>
    public string Text => $"{ByteFormat.Format(Usage.UsedBytes)} / {ByteFormat.Format(Usage.TotalBytes)}" + (Usage.IsExact ? "" : " (this app)");
}
