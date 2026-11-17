using Microsoft.Extensions.Options;
using Sharplight.Runtime;

namespace Sharplight.Hosting;

/// <summary>
/// The devices of the available execution providers (GPUs first) and the pool they are opened from: each physical device is opened
/// once (with the host's budget and disabled features) and shared by every model on it. Thread-safe; dispose to close the devices
/// (after the model host that uses them).
/// </summary>
public sealed class DeviceInventory : IDisposable
{
    private readonly ExecutionProviderCatalog _providers;
    private readonly ModelHostingOptions _options;
    private readonly DevicePool _pool = new();
    private readonly Lazy<IReadOnlyList<HostDevice>> _devices;

    /// <summary>Creates the inventory.</summary>
    /// <param name="providers">The execution providers (<c>services.AddSharplight()</c>).</param>
    /// <param name="options">The hosting options (budget and disabled features of opened GPUs).</param>
    public DeviceInventory(ExecutionProviderCatalog providers, IOptions<ModelHostingOptions> options)
    {
        ArgumentNullException.ThrowIfNull(providers);
        ArgumentNullException.ThrowIfNull(options);
        _providers = providers;
        _options = options.Value;
        _devices = new(Discover);
    }

    /// <summary>Every device of every available provider, GPUs first; providers that fail to enumerate are skipped.</summary>
    public IReadOnlyList<HostDevice> Devices => _devices.Value;

    /// <summary>The devices opened so far.</summary>
    public IReadOnlyList<IComputeDevice> OpenedDevices => _pool.Devices;

    /// <summary>The CPU device, or null when the CPU provider is not available.</summary>
    public HostDevice? Cpu => Devices.FirstOrDefault(device => !device.IsGpu);

    /// <summary>Opens <paramref name="device"/> (once; later calls return the same instance).</summary>
    /// <param name="device">The device.</param>
    /// <returns>The opened device, owned by the inventory.</returns>
    public IComputeDevice Open(HostDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        var options = new DeviceOptions { MemoryBudget = device.IsGpu ? _options.Budget : null };
        foreach (var feature in _options.DisabledFeatures)
        {
            options.DisabledFeatures.Add(feature);
        }

        return _pool.Open(device.Provider, device.Info, options);
    }

    /// <summary>The opened device of <paramref name="device"/>, or null when it was not opened yet.</summary>
    /// <param name="device">The device.</param>
    public IComputeDevice? FindOpened(HostDevice device)
    {
        ArgumentNullException.ThrowIfNull(device);
        return _pool.Devices.FirstOrDefault(opened => opened.Info.Provider == device.Info.Provider && opened.Info.Index == device.Info.Index);
    }

    /// <summary>
    /// The devices a specification names (<see cref="DeviceSpecifications"/>), in order; <see cref="DeviceSpecifications.Auto"/> entries
    /// are kept as null (the placement policy chooses them).
    /// </summary>
    /// <param name="specification">The specification.</param>
    /// <returns>One entry per part; null for <c>auto</c>.</returns>
    /// <exception cref="InvalidOperationException">A part names no available device.</exception>
    public IReadOnlyList<HostDevice?> Resolve(string? specification)
    {
        return [.. DeviceSpecifications.Split(specification).Select(ResolveOne)];
    }

    private HostDevice? ResolveOne(string part)
    {
        if (part.Equals(DeviceSpecifications.Auto, StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        if (part.Equals(DeviceSpecifications.Cpu, StringComparison.OrdinalIgnoreCase))
        {
            return Cpu ?? throw new InvalidOperationException("The CPU provider is not available.");
        }

        if (int.TryParse(part, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int index))
        {
            var gpus = Devices.Where(device => device.IsGpu).ToList();
            var provider = gpus.FirstOrDefault()?.Provider.Name;
            var ofProvider = gpus.Where(device => device.Provider.Name == provider).ToList();
            return index < ofProvider.Count
                ? ofProvider[index]
                : throw new InvalidOperationException(gpus.Count == 0
                    ? $"Device '{part}': no GPU is available (use 'cpu')."
                    : $"Device '{part}': provider '{provider}' has {ofProvider.Count} device(s).");
        }

        return Devices.FirstOrDefault(device => device.Id.Equals(part, StringComparison.OrdinalIgnoreCase))
            ?? throw new InvalidOperationException($"Device '{part}' is neither 'auto', 'cpu', a GPU index nor an available device id ({string.Join(", ", Devices.Select(device => device.Id))}).");
    }

    /// <summary>Closes every opened device.</summary>
    public void Dispose() => _pool.Dispose();

    private List<HostDevice> Discover()
    {
        var found = new List<HostDevice>();
        foreach (var (provider, availability) in _providers.Discover())
        {
            if (!availability.IsAvailable)
            {
                continue;
            }

            IReadOnlyList<DeviceInfo> devices;
            try
            {
                devices = provider.GetDevices();
            }
            catch (Exception)
            {
                continue;
            }

            found.AddRange(devices.Select(info => new HostDevice($"{info.Provider}:{info.Index}", info.Name, provider, info)));
        }

        return [.. found.OrderBy(device => device.IsGpu ? 0 : 1)];
    }
}
