using System.Runtime.CompilerServices;
using Sharplight.Distributed.Transport;
using Sharplight.Runtime;

namespace Sharplight.Distributed;

/// <summary>
/// The execution provider whose devices are segment workers on other machines (<see cref="RemoteSegmentOptions.Workers"/>, in pipeline
/// order). A placement lists them after or instead of local devices (<see cref="PlacementOptions.Devices"/>): the graph is split into
/// per-device segments exactly as over local GPUs, and the segments placed on a worker are compiled and run there. Listing the devices
/// connects to every worker and reads its devices and memory, so placement weights each worker by its free memory.
/// </summary>
/// <remarks>
/// The provider is not meant for automatic device selection (hosts list remote devices explicitly); its devices report
/// <see cref="DeviceKind.Other"/>.
/// </remarks>
public sealed class RemoteExecutionProvider : IExecutionProvider
{
    private readonly RemoteSegmentOptions _options;
    private readonly ISegmentTransport _transport;
    private readonly IReadOnlyList<SegmentEndpoint> _workers;
    private readonly ConditionalWeakTable<object, RemotePipeline> _pipelines = [];
    private readonly Dictionary<int, WorkerConnection> _listed = [];
    private readonly Lock _lock = new();
    private IReadOnlyList<DeviceInfo>? _devices;
    private bool _disposed;

    /// <summary>Creates the provider; nothing connects until <see cref="GetDevices"/> or <see cref="OpenDevice"/>.</summary>
    /// <param name="options">The workers, transport and timeouts.</param>
    /// <exception cref="FormatException">A worker address is not <c>host:port</c>.</exception>
    public RemoteExecutionProvider(RemoteSegmentOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _options = options;
        _transport = options.ResolvedTransport;
        _workers = [.. options.Workers.Select(SegmentEndpoint.Parse)];
    }

    /// <summary>The provider name in device ids (<c>remote:0</c>).</summary>
    public const string ProviderName = "remote";

    /// <inheritdoc/>
    public string Name => ProviderName;

    /// <summary>The workers in pipeline order.</summary>
    public IReadOnlyList<SegmentEndpoint> Workers => _workers;

    /// <inheritdoc/>
    public string? PortableCodeFormat => null;

    /// <summary>Available when workers are configured (whether they answer shows when the devices are listed).</summary>
    public ProviderAvailability CheckAvailability() => _workers.Count > 0
        ? ProviderAvailability.Available($"{_workers.Count} segment worker(s)")
        : ProviderAvailability.Unavailable("no segment workers are configured");

    /// <summary>Connects to every worker (once) and describes each as one device.</summary>
    /// <exception cref="SegmentWorkerException">A worker cannot be reached or refused the connection.</exception>
    public IReadOnlyList<DeviceInfo> GetDevices()
    {
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_devices is not null)
            {
                return _devices;
            }

            var devices = new List<DeviceInfo>();
            for (int i = 0; i < _workers.Count; i++)
            {
                var connection = WorkerConnection.OpenAsync(_transport, _workers[i], _options, CancellationToken.None).GetAwaiter().GetResult();
                _listed[i] = connection;
                devices.Add(Describe(i, connection));
            }

            return _devices = devices;
        }
    }

    /// <summary>
    /// Opens a worker as a device (the connection made while listing, or a new one). The device owns the connection; disposing it makes the
    /// worker release what was compiled on it.
    /// </summary>
    /// <exception cref="ArgumentException">The device is not one of this provider's.</exception>
    /// <exception cref="SegmentWorkerException">The worker cannot be reached.</exception>
    public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        if (device.Provider != Name || device.Index < 0 || device.Index >= _workers.Count)
        {
            throw new ArgumentException($"Device {device} is not a device of the remote provider.", nameof(device));
        }

        WorkerConnection? connection;
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_listed.Remove(device.Index, out connection) && connection.Fault is not null)
            {
                connection.Dispose();
                connection = null;
            }
        }

        connection ??= WorkerConnection.OpenAsync(_transport, _workers[device.Index], _options, CancellationToken.None).GetAwaiter().GetResult();
        return new RemoteComputeDevice(connection, Describe(device.Index, connection), _options, _pipelines);
    }

    /// <summary>Connects to the workers and opens every one as a device, in pipeline order (the slots to append to a placement).</summary>
    /// <exception cref="SegmentWorkerException">A worker cannot be reached.</exception>
    public IReadOnlyList<RemoteComputeDevice> OpenAll() => [.. GetDevices().Select(d => (RemoteComputeDevice)OpenDevice(d))];

    /// <inheritdoc/>
    public bool CanCompileFor(DeviceProfile profile) => false;

    /// <summary>Always an empty set: workers compile their segments' kernels themselves.</summary>
    public Task<PortableCodeSet> CompilePortableAsync(PortableCompilationRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        return Task.FromResult(PortableCodeSet.Empty(Name));
    }

    /// <summary>Always an empty set: workers compile their segments' kernels themselves.</summary>
    public Task<DeviceCodeSet> CompileDeviceSpecificAsync(IComputeDevice device, PortableCodeSet portable, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(portable);
        return Task.FromResult(DeviceCodeSet.Empty(Name, device.Info.Name));
    }

    /// <summary>Closes the connections made while listing that no device took over.</summary>
    public void Dispose()
    {
        List<WorkerConnection> connections;
        lock (_lock)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            connections = [.. _listed.Values];
            _listed.Clear();
        }

        foreach (var connection in connections)
        {
            connection.Dispose();
        }
    }

    private DeviceInfo Describe(int index, WorkerConnection connection) => RemoteComputeDevice.Describe(Name, index, connection, _transport.Name);
}
