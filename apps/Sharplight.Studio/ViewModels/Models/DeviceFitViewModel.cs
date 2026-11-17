using Sharplight.Hosting.Fit;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>One device's row in a fit estimate: the model's part against what is used, free and reserved.</summary>
internal sealed class DeviceFitViewModel
{
    public DeviceFitViewModel(DeviceFit fit, bool offloaded = false)
    {
        Fit = fit;
        IsOffload = offloaded;
    }

    /// <summary>The figures.</summary>
    public DeviceFit Fit { get; }

    /// <summary>The row is the CPU's share of an offloaded model.</summary>
    public bool IsOffload { get; }

    /// <summary>Device name, or <c>System memory</c> for the CPU.</summary>
    public string Name => Fit.Device.IsGpu ? Fit.Device.Name : "System memory";

    /// <summary>The part fits.</summary>
    public bool Fits => Fit.Fits;

    /// <summary>The part does not fit.</summary>
    public bool IsOver => !Fit.Fits;

    /// <summary><c>Needs 9.1 GB</c>, or <c>Offloaded 3.2 GB</c> for the CPU's share.</summary>
    public string NeedText => (IsOffload ? "Offloaded " : "Needs ") + ByteFormat.Format(Fit.NeededBytes);

    /// <summary><c>13.9 GB free of 24 GB · 1 GB reserve</c> (with what unloading frees).</summary>
    public string MemoryText
    {
        get
        {
            var text = $"{ByteFormat.Format(Fit.Device.FreeBytes)} free of {ByteFormat.Format(Fit.Device.TotalBytes)} · {ByteFormat.Format(Fit.ReserveBytes)} reserve";
            if (Fit.FreedBytes > 0)
            {
                text += $" · +{ByteFormat.Format(Fit.FreedBytes)} from unloading";
            }

            return Fit.Device.IsExact ? text : text + " (this app's tally)";
        }
    }

    /// <summary>Fraction of the device used by everything else (after the unloads the plan counts on).</summary>
    public double UsedFraction => Fraction(Used);

    /// <summary>Fraction used once the model is loaded too.</summary>
    public double NeedFraction => Fraction(Used + Fit.NeededBytes);

    /// <summary>Fraction where the reserve starts.</summary>
    public double ReserveFraction => Fraction(Fit.Device.TotalBytes - Fit.ReserveBytes);

    private long Used => Math.Max(0, Fit.Device.TotalBytes - Fit.Device.FreeBytes - Fit.FreedBytes);

    private double Fraction(long bytes) => Fit.Device.TotalBytes <= 0 ? 0 : Math.Clamp((double)bytes / Fit.Device.TotalBytes, 0, 1);
}
