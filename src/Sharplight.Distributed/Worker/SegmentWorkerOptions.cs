using Sharplight.Distributed.Transport;
using Sharplight.Runtime;

namespace Sharplight.Distributed.Worker;

/// <summary>Settings of a <see cref="SegmentWorker"/>: where it listens and the local devices its segments run on.</summary>
public sealed class SegmentWorkerOptions
{
    /// <summary>Address to listen on, <c>host:port</c> (<c>0.0.0.0:7070</c> for every interface; port 0 picks a free one). Default <c>0.0.0.0:7070</c>.</summary>
    public string Listen { get; set; } = "0.0.0.0:7070";

    /// <summary>
    /// Address neighbouring workers use to reach this one, when it differs from the address the coordinator connects to (NAT, a separate
    /// data network); null (default) uses the coordinator's.
    /// </summary>
    public string? Advertise { get; set; }

    /// <summary>The worker's name in messages and status. Default: the machine name.</summary>
    public string Name { get; set; } = Environment.MachineName;

    /// <summary>The local devices in pipeline order: each segment a coordinator sends is placed over them like a model over local GPUs.</summary>
    public IList<DeviceSlot> Devices { get; set; } = [];

    /// <summary>Fallback device (the CPU) for layers that do not fit and ops the devices lack (CPU offload), or null.</summary>
    public DeviceSlot? Fallback { get; set; }

    /// <summary>Bytes kept free on every device for activations and other applications (null: the placement default, 1 GiB).</summary>
    public long? ReservePerDevice { get; set; }

    /// <summary>
    /// With a CPU <see cref="Fallback"/>: leave the routed experts that do not fit in memory on disk and read the ones each pass selects into
    /// a cache (docs/hardware.md "Routed experts on disk").
    /// </summary>
    public bool ExpertsOnDisk { get; set; }

    /// <summary>The expert cache size in bytes with <see cref="ExpertsOnDisk"/> (null: what host memory allows).</summary>
    public long? ExpertCacheBytes { get; set; }

    /// <summary>How the worker accepts connections and reaches its neighbours. Default: <see cref="TcpSegmentTransport"/> without TLS.</summary>
    public ISegmentTransport? Transport { get; set; }

    /// <summary>A coordinator silent this long (no request, no heartbeat) is considered gone; its segments are released. Default 15 s.</summary>
    public TimeSpan HeartbeatTimeout { get; set; } = TimeSpan.FromSeconds(15);

    /// <summary>Longest a connection handshake (and a neighbour's connect) may take. Default 10 s.</summary>
    public TimeSpan ConnectTimeout { get; set; } = TimeSpan.FromSeconds(10);

    /// <summary>Receives log lines (connections, compilations, failures), or null.</summary>
    public Action<string>? Log { get; set; }
}
