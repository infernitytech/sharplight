using System.Collections.Concurrent;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using Sharplight.Distributed.Protocol;
using Sharplight.Distributed.Transport;
using Sharplight.Graph;
using Sharplight.Package;
using Sharplight.Runtime;

namespace Sharplight.Distributed;

/// <summary>
/// A segment worker seen as one device of a placement (<see cref="DeviceSlot"/>): its memory is the worker's (its pipeline devices plus its
/// CPU fallback when it offloads), so placement weights it by free memory like a local GPU; <see cref="Compile"/> sends the segment to the
/// worker, which compiles it on its own devices and keeps its weights and KV cache. Thread-safe.
/// </summary>
public sealed class RemoteComputeDevice : IComputeDevice
{
    private readonly WorkerConnection _connection;
    private readonly ConditionalWeakTable<object, RemotePipeline> _pipelines;
    private readonly ConcurrentDictionary<string, OpSupport> _support = new(StringComparer.Ordinal);
    private readonly ConditionalWeakTable<RemotePipeline, object> _watched = [];
    private readonly Lock _statisticsLock = new();
    private long _executions;
    private double _roundTripMs;
    private double _workerMs;
    private double _waitMs;
    private long _bytesSentBase;
    private long _bytesReceivedBase;

    internal RemoteComputeDevice(WorkerConnection connection, DeviceInfo info, RemoteSegmentOptions options, ConditionalWeakTable<object, RemotePipeline> pipelines)
    {
        _connection = connection;
        _pipelines = pipelines;
        Info = info;
        Options = options;
    }

    /// <inheritdoc/>
    public DeviceInfo Info { get; }

    /// <summary>The worker's address.</summary>
    public SegmentEndpoint Endpoint => _connection.Endpoint;

    /// <summary>What the worker reported when the connection was opened.</summary>
    public SegmentWorkerInfo Worker => _connection.Info;

    /// <summary>Why the connection to the worker failed, or null while it works.</summary>
    public SegmentWorkerException? Fault => _connection.Fault;

    /// <summary>The provider's settings.</summary>
    internal RemoteSegmentOptions Options { get; }

    /// <summary>Executions, round-trip and worker times, and bytes on the wire since the connection opened or <see cref="ResetStatistics"/>.</summary>
    public RemoteSegmentStatistics Statistics
    {
        get
        {
            lock (_statisticsLock)
            {
                return new RemoteSegmentStatistics(_executions, _roundTripMs, _workerMs, _waitMs, _connection.BytesSent - _bytesSentBase, _connection.BytesReceived - _bytesReceivedBase);
            }
        }
    }

    /// <inheritdoc/>
    public long AvailableMemory => GetMemoryInfo().Available;

    /// <inheritdoc/>
    public IReadOnlySet<string> EnabledFeatures => DeviceFeatures.Effective(Info.Features);

    /// <summary>The worker's live figures: the sum over its devices (its CPU fallback included when it offloads).</summary>
    /// <exception cref="SegmentWorkerException">The worker does not answer.</exception>
    public DeviceMemoryInfo GetMemoryInfo()
    {
        var reply = _connection.Call(FrameKind.MemoryRequest, [], Options.ExecuteTimeout);
        var r = new PayloadReader(reply.Payload);
        int count = r.Count();
        long total = 0;
        long available = 0;
        for (int i = 0; i < count; i++)
        {
            total += r.Int64();
            available += r.Int64();
        }

        return new DeviceMemoryInfo(total, total, Math.Max(0, total - available), 0, true);
    }

    /// <summary>Asks the worker whether its devices run the node (answers are cached per op, attributes and operand types).</summary>
    public OpSupport Supports(Node node)
    {
        ArgumentNullException.ThrowIfNull(node);
        return _support.GetOrAdd(SupportKey(node), _ => AskSupport(node));
    }

    /// <summary>
    /// Sends the segment to the worker and waits until it is compiled there. The worker checks its weights against the coordinator's
    /// (<see cref="RemoteSegmentOptions.WeightVerification"/>), places the segment over its own devices and keeps the weights and state.
    /// </summary>
    /// <exception cref="SegmentWorkerException">The worker refused or failed the compilation (another model copy, out of memory, ...), or is gone.</exception>
    /// <exception cref="NotSupportedException">The segment holds control flow or exchanges strings.</exception>
    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options)
    {
        ArgumentNullException.ThrowIfNull(segment);
        ArgumentNullException.ThrowIfNull(options);
        var pipeline = options.StateScope is { } scope ? _pipelines.GetValue(scope, _ => new RemotePipeline()) : new RemotePipeline();
        pipeline.ThrowIfFailed();
        Watch(pipeline);
        var verification = Options.WeightVerification;
        var fingerprint = SegmentFingerprint.Compute(segment.Weights.Select(v => v.Weight!), verification);
        using var blob = new MemoryStream();
        using (var bw = new BinaryWriter(blob, Encoding.UTF8, leaveOpen: true))
        {
            SegmentCodec.Write(bw, segment);
        }

        var w = new PayloadWriter((int)blob.Length + 1024);
        CompileOptionsCodec.Write(w, options);
        w.Byte((byte)verification).String(fingerprint);
        var precision = Options.BoundaryPrecision;
        var narrowInputs = segment.Inputs.Select(v => BoundaryCodec.IsNarrowed(v, segment.Graph, precision)).ToArray();
        var narrowOutputs = segment.Outputs.Select(v => BoundaryCodec.IsNarrowed(v, segment.Graph, precision)).ToArray();
        w.Bytes(blob.GetBuffer().AsSpan(0, (int)blob.Length));
        w.Byte((byte)precision).Indices(narrowInputs).Indices(narrowOutputs);
        Frame reply;
        try
        {
            reply = _connection.Call(FrameKind.Compile, w.WrittenSpan, Options.CompileTimeout, options.CancellationToken);
        }
        catch (SegmentWorkerException e) when (!e.WorkerAlive)
        {
            throw pipeline.Fail(e);
        }

        var r = new PayloadReader(reply.Payload);
        uint handle = r.UInt32();
        long bytes = r.Int64();
        bool canCopy = r.Bool();
        return new RemoteCompiledSegment(this, _connection, pipeline, segment, handle, bytes, canCopy, ProducedSymbols(segment))
        {
            Precision = precision,
            NarrowInputs = narrowInputs,
            NarrowOutputs = narrowOutputs,
        };
    }

    /// <summary>Starts <see cref="Statistics"/> again from zero executions .</summary>
    public void ResetStatistics()
    {
        lock (_statisticsLock)
        {
            _executions = 0;
            _roundTripMs = 0;
            _workerMs = 0;
            _waitMs = 0;
            _bytesSentBase = _connection.BytesSent;
            _bytesReceivedBase = _connection.BytesReceived;
        }
    }

    /// <summary>Adds an execution to <see cref="Statistics"/>.</summary>
    internal void Record(double roundTripMs, double workerMs, double waitMs)
    {
        lock (_statisticsLock)
        {
            _executions++;
            _roundTripMs += roundTripMs;
            _workerMs += workerMs;
            _waitMs += waitMs;
        }
    }

    /// <summary>Fails <paramref name="pipeline"/> when the connection fails, so that segments on other workers stop waiting.</summary>
    private void Watch(RemotePipeline pipeline)
    {
        pipeline.Add(_connection);
        lock (_watched)
        {
            if (_watched.TryGetValue(pipeline, out _))
            {
                return;
            }

            _watched.Add(pipeline, pipeline);
        }

        var weak = new WeakReference<RemotePipeline>(pipeline);
        _connection.Faulted += fault =>
        {
            if (weak.TryGetTarget(out var target))
            {
                target.Fail(fault);
            }
        };
    }

    private OpSupport AskSupport(Node node)
    {
        using var blob = new MemoryStream();
        using (var bw = new BinaryWriter(blob, Encoding.UTF8, leaveOpen: true))
        {
            try
            {
                SegmentCodec.WriteNode(bw, node);
            }
            catch (NotSupportedException e)
            {
                return OpSupport.No(e.Message);
            }
        }

        var reply = _connection.Call(FrameKind.SupportsRequest, blob.GetBuffer().AsSpan(0, (int)blob.Length), Options.ExecuteTimeout);
        var r = new PayloadReader(reply.Payload);
        return r.Bool() ? OpSupport.Yes : OpSupport.No(r.String());
    }

    /// <summary>What decides support: the op, its attributes and the kind, role, type and shape of every operand and result.</summary>
    private static string SupportKey(Node node)
    {
        var key = new StringBuilder(node.Op.Name).Append('|').Append(node.Attributes);
        foreach (var v in node.Inputs)
        {
            key.Append('|').Append(v.Kind).Append(':').Append(v.Input?.Role).Append(':').Append(v.Type).Append(v.Shape);
        }

        key.Append("|w").Append(node.Writes.Length);
        foreach (var v in node.Outputs)
        {
            key.Append("|o").Append(v.Type).Append(v.Shape);
        }

        return key.ToString();
    }

    /// <summary>The data-dependent symbols the segment's ops may produce (later segments need them as soon as it ran).</summary>
    private static List<string> ProducedSymbols(GraphSegment segment)
    {
        var dynamic = segment.Graph.DynamicSymbols.Select(kv => kv.Key).ToList();
        if (dynamic.Count == 0)
        {
            return [];
        }

        return [.. dynamic.Where(symbol => segment.Nodes.Any(n => n.Outputs.Any(o => o.Shape.Dims.Any(d => d.IsSymbolic && d.Symbol!.Contains(symbol, StringComparison.Ordinal)))))];
    }

    /// <summary>The device description of a connected worker.</summary>
    internal static DeviceInfo Describe(string provider, int index, WorkerConnection connection, string transport)
    {
        var info = connection.Info;
        var devices = info.Devices;
        return new DeviceInfo
        {
            Provider = provider,
            Index = index,
            Name = $"{connection.Endpoint} ({string.Join(" + ", devices.Select(d => d.IsFallback ? $"{d.Name} fallback" : d.Name))})",
            Kind = DeviceKind.Other,
            TotalMemory = devices.Sum(d => d.TotalMemory),
            Driver = $"sharplight segment worker {info.Version}",
            Uuid = info.InstanceId,
            Features = info.Features,
            Capabilities = new Dictionary<string, string>
            {
                ["endpoint"] = connection.Endpoint.ToString(),
                ["worker"] = info.Name,
                ["model"] = info.ModelName,
                ["architecture"] = info.Architecture,
                ["devices"] = devices.Count.ToString(CultureInfo.InvariantCulture),
                ["transport"] = transport,
            },
        };
    }

    /// <summary>Closes the connection; the worker releases everything compiled over it.</summary>
    public void Dispose() => _connection.Dispose();
}
