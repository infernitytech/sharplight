using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Services.Fit;

/// <summary>The part of a model one device would hold, against its memory.</summary>
/// <param name="Device">The device and its memory.</param>
/// <param name="NeededBytes">Bytes of the model on this device.</param>
/// <param name="FreedBytes">Bytes freed on it by unloading the models the plan names.</param>
/// <param name="ReserveBytes">Bytes kept free for other applications.</param>
internal sealed record DeviceFit(DeviceMemory Device, long NeededBytes, long FreedBytes, long ReserveBytes)
{
    /// <summary>What the model may use: free memory plus what unloading frees, minus the reserve.</summary>
    public long AvailableBytes => Math.Max(0, Device.FreeBytes + FreedBytes - ReserveBytes);

    /// <summary>The part fits.</summary>
    public bool Fits => NeededBytes <= AvailableBytes;
}
