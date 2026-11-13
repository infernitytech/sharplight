namespace Sharplight.Runtime;

/// <summary>
/// A backend (Vulkan, CUDA, CPU, Metal, ...). Providers are cheap to construct; they only touch the
/// driver when <see cref="CheckAvailability"/> / <see cref="GetDevices"/> are called.
/// </summary>
public interface IExecutionProvider : IDisposable
{
    /// <summary>Provider name (for example "vulkan" or "cpu").</summary>
    public string Name { get; }
    /// <summary>Detects whether the provider can run on this machine (driver/library present, ...).</summary>
    public ProviderAvailability CheckAvailability();
    /// <summary>Lists the devices; empty when the provider is unavailable.</summary>
    public IReadOnlyList<DeviceInfo> GetDevices();
    /// <summary>
    /// Opens a device for compiling and running segments; the caller disposes it.
    /// </summary>
    /// <param name="device">A device from <see cref="GetDevices"/>.</param>
    /// <param name="options">Device options, or <see langword="null"/> for the defaults.</param>
    /// <exception cref="NotSupportedException">The provider is unavailable.</exception>
    public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null);
}
