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

    /// <summary>The <paramref name="index"/>-th device of the first available provider named <paramref name="providerName"/>.</summary>
    /// <param name="providers">The providers to choose from.</param>
    /// <param name="providerName">The provider name, e.g. <c>vulkan</c>.</param>
    /// <param name="index">The device index within the provider.</param>
    /// <param name="options">Device options for the first open, or <see langword="null"/>.</param>
    /// <exception cref="InvalidOperationException">No available provider has this name.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The provider has no device with this index.</exception>
    public IComputeDevice Open(ExecutionProviderCatalog providers, string providerName, int index = 0, DeviceOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(providers);
        if (!providers.TryFind(providerName, out var provider))
        {
            throw new InvalidOperationException($"No available execution provider named '{providerName}'.");
        }

        var devices = provider.GetDevices();
        if (index < 0 || index >= devices.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(index), $"Provider '{providerName}' has {devices.Count} device(s).");
        }

        return Open(provider, devices[index], options);
    }

    /// <summary>Snapshot of the devices opened so far.</summary>
    public IReadOnlyList<IComputeDevice> Devices
    {
        get
        {
            lock (_lock)
            {
                return [.. _devices.Values];
            }
        }
    }

    /// <summary>Disposes every device that was opened through the pool.</summary>
    public void Dispose()
    {
        List<IComputeDevice> all;
        lock (_lock)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            all = [.. _devices.Values];
            _devices.Clear();
        }

        foreach (var d in all)
        {
            d.Dispose();
        }
    }
}
