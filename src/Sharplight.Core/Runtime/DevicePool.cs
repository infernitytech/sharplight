namespace Sharplight.Runtime;

/// <summary>
/// Opens each physical device once and hands the same <see cref="IComputeDevice"/> to every model that asks for it, so
/// several models share one logical device (one VkDevice, queue, pipeline cache and memory allocator per GPU) and
/// their memory is accounted for in one place. Devices are owned by the pool: dispose the pool, not the devices.
/// Opened devices are safe to use from several threads (each model runs its own scheduler thread).
/// </summary>
public sealed class DevicePool : IDisposable
{
    private readonly Dictionary<string, IComputeDevice> _devices = [];
    private readonly Lock _lock = new();
    private bool _disposed;

    /// <summary>
    /// Opens <paramref name="device"/> on <paramref name="provider"/>, or returns the device already opened for it.
    /// <paramref name="options"/> only apply to the first open; per-model tweaks belong in compile options
    /// (<see cref="CompileOptions.DisabledFeatures"/>).
    /// </summary>
    public IComputeDevice Open(IExecutionProvider provider, DeviceInfo device, DeviceOptions? options = null)
    {
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var key = $"{device.Provider}:{device.Index}";
            if (!_devices.TryGetValue(key, out var d))
            {
                _devices[key] = d = provider.OpenDevice(device, options);
            }

            return d;
        }
    }

    /// <summary>Convenience: the <paramref name="index"/>-th device of the first available provider named <paramref name="providerName"/>.</summary>
    public IComputeDevice Open(string providerName, int index = 0, DeviceOptions? options = null)
    {
        foreach (var (p, a) in ProviderRegistry.Discover())
        {
            if (!a.IsAvailable || p.Name != providerName)
            {
                continue;
            }

            var devs = p.GetDevices();
            if (index < 0 || index >= devs.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(index), $"Provider '{providerName}' has {devs.Count} device(s).");
            }

            return Open(p, devs[index], options);
        }
        throw new InvalidOperationException($"No available execution provider named '{providerName}'.");
    }

    public IReadOnlyList<IComputeDevice> Devices { get { lock (_lock) { return [.. _devices.Values]; } } }

    public void Dispose()
    {
        List<IComputeDevice> all;
        lock (_lock) { if (_disposed) { return; } _disposed = true; all = [.. _devices.Values]; _devices.Clear(); }
        foreach (var d in all)
        {
            d.Dispose();
        }
    }
}
