using System.Collections.Concurrent;
using System.Reflection;
using Sharplight.Distributed.Protocol;
using Sharplight.Distributed.Transport;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;

namespace Sharplight.Distributed.Worker;

/// <summary>
/// A segment worker: serves the layers of one model to coordinators on other machines (multi-node pipeline parallel inference). It holds
/// its own copy of the model (any format the loader reads: .slm, GGUF, ...); each coordinator connection compiles segments of that model on
/// the worker's devices (<see cref="SegmentWorkerOptions.Devices"/>, CPU offload and experts on disk included), so only the layers placed
/// here are read and uploaded, and the KV cache and recurrent state of those layers live here. Boundary activations arrive with each
/// execution or straight from the neighbouring worker, and go on to the next one. Everything a connection compiled is released when it
/// closes or goes silent.
/// </summary>
/// <example>
/// <code>
/// var model = services.GetRequiredService&lt;ModelLoader&gt;().Build("qwen3-8b-q4_k_m.gguf");
/// using var worker = new SegmentWorker(model, new SegmentWorkerOptions { Listen = "0.0.0.0:7070", Devices = [new DeviceSlot { Device = gpu }] });
/// worker.Start();
/// await worker.Completion;
/// </code>
/// </example>
public sealed class SegmentWorker : IDisposable
{
    private readonly ModelDefinition _model;
    private readonly SegmentWorkerOptions _options;
    private readonly ISegmentTransport _transport;
    private readonly Dictionary<string, WeightInfo> _weights;
    private readonly ConcurrentDictionary<WorkerSession, byte> _sessions = new();
    private readonly CancellationTokenSource _stopping = new();
    private readonly TaskCompletionSource _completion = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private ISegmentListener? _listener;
    private int _disposed;

    /// <summary>Creates the worker; nothing listens until <see cref="Start"/>.</summary>
    /// <param name="model">The worker's copy of the model (its weights are read only for the layers placed here).</param>
    /// <param name="options">Address, devices and timeouts.</param>
    /// <exception cref="ArgumentException">No device is configured.</exception>
    public SegmentWorker(ModelDefinition model, SegmentWorkerOptions options)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(options);
        if (options.Devices.Count == 0)
        {
            throw new ArgumentException("A segment worker needs at least one device.", nameof(options));
        }

        _model = model;
        _options = options;
        _transport = options.Transport ?? new TcpSegmentTransport();
        _weights = new Dictionary<string, WeightInfo>(model.Weights, StringComparer.Ordinal);
        foreach (var weight in model.Graphs.Values.SelectMany(DefinitionCodec.WithSubgraphs).SelectMany(g => g.Weights).Select(v => v.Weight!))
        {
            _weights.TryAdd(weight.Name, weight);
        }

        Mailbox = new PeerMailbox();
        Links = new PeerLinks(_transport, options.ConnectTimeout, options.Name);
        DiskExperts = options.ExpertsOnDisk && options.Fallback is { Device.Info.Kind: DeviceKind.Cpu }
            ? new Runtime.Experts.ExpertStore(options.ExpertCacheBytes, log: options.Log)
            : null;
        InstanceId = Guid.NewGuid().ToString("N");
    }

    /// <summary>The bound address once started (with the port the system chose when 0 was asked for).</summary>
    /// <exception cref="InvalidOperationException">The worker is not started.</exception>
    public SegmentEndpoint Endpoint => _listener?.LocalEndpoint ?? throw new InvalidOperationException("The segment worker is not started.");

    /// <summary>Completes when the worker stops (after <see cref="Dispose"/>), or faults when accepting connections fails.</summary>
    public Task Completion => _completion.Task;

    /// <summary>Coordinator connections currently served.</summary>
    public int Connections => _sessions.Count;

    /// <summary>Random id of this worker process (coordinators use it to recognise two connections to the same process).</summary>
    public string InstanceId { get; }

    internal SegmentWorkerOptions Options => _options;

    internal PeerMailbox Mailbox { get; }

    internal PeerLinks Links { get; }

    internal SharedWeights SharedWeights { get; } = new();

    internal Runtime.Experts.ExpertStore? DiskExperts { get; }

    /// <summary>Starts listening and accepting connections.</summary>
    /// <exception cref="FormatException"><see cref="SegmentWorkerOptions.Listen"/> is not <c>host:port</c>.</exception>
    /// <exception cref="InvalidOperationException">The worker was started already.</exception>
    public void Start()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_listener is not null)
        {
            throw new InvalidOperationException("The segment worker is started already.");
        }

        _listener = _transport.Listen(SegmentEndpoint.Parse(_options.Listen));
        Log($"segment worker {_options.Name} listening on {_listener.LocalEndpoint} ({_transport.Name}), model {_model.Name} ({_model.Architecture}), "
            + string.Join(", ", Describe().Select(d => $"{d.Name}{(d.IsFallback ? " (fallback)" : "")} {d.AvailableMemory >> 20} MiB free")));
        _ = AcceptLoop(_listener);
    }

    /// <summary>The worker's devices and model for a coordinator's handshake.</summary>
    internal SegmentWorkerInfo Info() => new()
    {
        Name = _options.Name,
        InstanceId = InstanceId,
        AdvertisedEndpoint = _options.Advertise is { } advertise ? SegmentEndpoint.Parse(advertise) : null,
        ModelName = _model.Name,
        Architecture = _model.Architecture,
        Devices = Describe(),
        Features = CommonFeatures(),
        Version = typeof(SegmentWorker).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "",
    };

    /// <summary>The pipeline devices, then the fallback, with their live memory figures.</summary>
    internal IReadOnlyList<SegmentWorkerDevice> Describe()
    {
        var devices = _options.Devices.Select(slot => Device(slot, isFallback: false)).ToList();
        if (_options.Fallback is { } fallback)
        {
            devices.Add(Device(fallback, isFallback: true));
        }

        return devices;
    }

    /// <summary>The features every pipeline device has enabled.</summary>
    private HashSet<string> CommonFeatures()
    {
        HashSet<string>? common = null;
        foreach (var slot in _options.Devices)
        {
            if (common is null)
            {
                common = new HashSet<string>(slot.Device.EnabledFeatures, StringComparer.Ordinal);
            }
            else
            {
                common.IntersectWith(slot.Device.EnabledFeatures);
            }
        }

        return common ?? [];
    }

    /// <summary>The worker's weight of that name (the model's, or one its graphs derive), or null.</summary>
    internal WeightInfo? Weight(string name) => _weights.GetValueOrDefault(name);

    internal void Log(string message) => _options.Log?.Invoke(message);

    internal void Remove(WorkerSession session) => _sessions.TryRemove(session, out _);

    private static SegmentWorkerDevice Device(DeviceSlot slot, bool isFallback)
    {
        var memory = slot.Device.GetMemoryInfo();
        long available = slot.MemoryLimit is { } limit ? Math.Min(limit, memory.Available) : memory.Available;
        long total = slot.MemoryLimit is { } cap ? Math.Min(cap, memory.Total) : memory.Total;
        var info = slot.Device.Info;
        return new SegmentWorkerDevice(info.Provider, info.Name, info.Kind, total, available, isFallback);
    }

    private async Task AcceptLoop(ISegmentListener listener)
    {
        try
        {
            while (!_stopping.IsCancellationRequested)
            {
                var connection = await listener.AcceptAsync(_stopping.Token).ConfigureAwait(false);
                _ = Task.Run(() => Serve(connection));
            }

            _completion.TrySetResult();
        }
        catch (OperationCanceledException)
        {
            _completion.TrySetResult();
        }
        catch (Exception e)
        {
            Log($"segment worker: accepting connections failed: {e.Message}");
            _completion.TrySetException(e);
        }
    }

    /// <summary>Runs the handshake of an accepted connection and serves it as a coordinator session or a neighbour's link.</summary>
    private void Serve(ISegmentConnection connection)
    {
        byte role;
        try
        {
            var hello = Task.Run(connection.Receive).WaitAsync(_options.ConnectTimeout).GetAwaiter().GetResult();
            var r = new PayloadReader(hello.Payload);
            if (hello.Kind != FrameKind.Hello || r.UInt32() != SegmentProtocol.Magic)
            {
                throw new InvalidDataException("not a Sharplight segment protocol handshake");
            }

            int version = r.Int32();
            role = r.Byte();
            string peer = r.String();
            if (version != SegmentProtocol.Version)
            {
                Reply(connection, FrameKind.Error, new PayloadWriter().Byte(SegmentProtocol.GeneralError)
                    .String($"protocol version {version} is not supported (this worker speaks {SegmentProtocol.Version})"));
                connection.Dispose();
                return;
            }

            var ack = new PayloadWriter().UInt32(SegmentProtocol.Magic).Int32(SegmentProtocol.Version);
            Info().Write(ack);
            connection.Send(FrameKind.HelloAck, 0, ack.WrittenSpan);
            Log($"{(role == SegmentProtocol.PeerRole ? "neighbour" : "coordinator")} {peer} connected from {connection.RemoteName}");
        }
        catch (Exception e)
        {
            Log($"handshake with {connection.RemoteName} failed: {e.Message}");
            connection.Dispose();
            return;
        }

        if (role == SegmentProtocol.PeerRole)
        {
            ReceivePeerValues(connection);
            return;
        }

        var session = new WorkerSession(this, connection);
        _sessions[session] = 0;
        if (Volatile.Read(ref _disposed) != 0)
        {
            session.Dispose();
            return;
        }

        session.Run();
    }

    /// <summary>Reads a neighbour's boundary values into the mailbox until the link closes.</summary>
    private void ReceivePeerValues(ISegmentConnection connection)
    {
        try
        {
            while (true)
            {
                var frame = connection.Receive();
                if (frame.Kind != FrameKind.PeerValue)
                {
                    continue;
                }

                var r = new PayloadReader(frame.Payload);
                long tag = r.Int64();
                long sequence = r.Int64();
                if (r.Bool())
                {
                    Mailbox.Post(tag, sequence, r.Bytes().ToArray(), null, connection);
                }
                else
                {
                    Mailbox.Post(tag, sequence, null, r.String(), connection);
                }
            }
        }
        catch (Exception e) when (e is IOException or InvalidDataException)
        {
            Mailbox.SourceClosed(connection, $"{connection.RemoteName}: {e.Message}");
            connection.Dispose();
        }
    }

    private static void Reply(ISegmentConnection connection, FrameKind kind, PayloadWriter payload)
    {
        try
        {
            connection.Send(kind, 0, payload.WrittenSpan);
        }
        catch (IOException)
        {
            // The other end is gone already.
        }
    }

    /// <summary>Stops listening, closes every connection (releasing what was compiled) and the links to neighbours.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _stopping.Cancel();
        _listener?.Dispose();
        foreach (var session in _sessions.Keys)
        {
            session.Dispose();
        }

        Links.Dispose();
        DiskExperts?.Dispose();
        _completion.TrySetResult();
    }
}
