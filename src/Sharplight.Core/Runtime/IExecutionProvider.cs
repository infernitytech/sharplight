namespace Sharplight.Runtime;

/// <summary>
/// A backend (Vulkan, CUDA, CPU, Metal, ...). Providers are cheap to construct; they only touch the
/// driver when <see cref="CheckAvailability"/> / <see cref="GetDevices"/> are called.
/// </summary>
public interface IExecutionProvider : IDisposable
{
    public string Name { get; }
    /// <summary>Detects whether the provider can run on this machine (driver/library present, ...).</summary>
    public ProviderAvailability CheckAvailability();
    public IReadOnlyList<DeviceInfo> GetDevices();
    public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null);
}
