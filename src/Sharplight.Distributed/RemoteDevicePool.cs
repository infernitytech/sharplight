using System.Runtime.CompilerServices;
using Microsoft.Extensions.Options;
using Sharplight.Distributed.Transport;

namespace Sharplight.Distributed;

/// <summary>
/// Opens each segment worker once and hands the same <see cref="RemoteComputeDevice"/> to every load that names it, like
/// <see cref="Runtime.DevicePool"/> for local devices; a worker whose connection failed is connected again on the next open (its segments
/// were released when the connection failed). Hosts register it with <c>services.AddRemoteSegments()</c>. Thread-safe.
/// </summary>
public sealed class RemoteDevicePool : IDisposable
{
    private readonly RemoteSegmentOptions _options;
    private readonly ISegmentTransport _transport;
    private readonly ConditionalWeakTable<object, RemotePipeline> _pipelines = [];
    private readonly Dictionary<SegmentEndpoint, RemoteComputeDevice> _devices = [];
    private readonly List<RemoteComputeDevice> _retired = [];
    private readonly Lock _lock = new();
    private int _nextIndex;
    private bool _disposed;

    /// <summary>Creates the pool with the transport and timeouts of <paramref name="options"/> (its worker list is not used).</summary>
    public RemoteDevicePool(IOptions<RemoteSegmentOptions> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _options = options.Value;
        _transport = _options.ResolvedTransport;
    }

    /// <summary>The worker at <paramref name="endpoint"/> as a device: the one opened before while its connection works, else a new connection.</summary>
    /// <exception cref="SegmentWorkerException">The worker cannot be reached or refused the connection.</exception>
    public RemoteComputeDevice Open(SegmentEndpoint endpoint)
    {
        ArgumentNullException.ThrowIfNull(endpoint);
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_devices.TryGetValue(endpoint, out var existing))
            {
                if (existing.Fault is null)
                {
                    return existing;
                }

                // Models compiled on the failed connection may still hold it; it is disposed with the pool.
                _retired.Add(existing);
                _devices.Remove(endpoint);
            }

            var connection = WorkerConnection.OpenAsync(_transport, endpoint, _options, CancellationToken.None).GetAwaiter().GetResult();
            var info = RemoteComputeDevice.Describe(RemoteExecutionProvider.ProviderName, _nextIndex++, connection, _transport.Name);
            var device = new RemoteComputeDevice(connection, info, _options, _pipelines);
            _devices[endpoint] = device;
            return device;
        }
    }

    /// <summary>The workers opened so far whose connections work.</summary>
    public IReadOnlyList<RemoteComputeDevice> Devices
    {
        get
        {
            lock (_lock)
            {
                return [.. _devices.Values.Where(d => d.Fault is null)];
            }
        }
    }

    /// <summary>Closes every connection; the workers release what was compiled on them.</summary>
    public void Dispose()
    {
        List<RemoteComputeDevice> devices;
        lock (_lock)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            devices = [.. _devices.Values, .. _retired];
            _devices.Clear();
            _retired.Clear();
        }

        foreach (var device in devices)
        {
            device.Dispose();
        }
    }
}
