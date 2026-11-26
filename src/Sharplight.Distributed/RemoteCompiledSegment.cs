using System.Diagnostics;
using Sharplight.Distributed.Protocol;
using Sharplight.Distributed.Transport;
using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Distributed;

/// <summary>
/// A segment compiled on a segment worker. Inputs set between executions travel with the next execution; the execution returns once it is
/// sent (the worker runs it while the coordinator goes on), and the first output read waits for the reply that carries the outputs. A boundary
/// value between two remote segments goes from worker to worker (<see cref="TryForwardFrom"/>), never through the coordinator.
/// </summary>
internal sealed class RemoteCompiledSegment : ICompiledSegment, IDeviceForwardingSegment
{
    /// <summary>Replies not read yet beyond which an execution waits for the oldest one (bounds the coordinator's memory).</summary>
    private const int MaxOutstanding = 64;

    private static readonly HashSet<Value> s_noSkips = [];
    private readonly RemoteComputeDevice _device;
    private readonly WorkerConnection _connection;
    private readonly RemotePipeline _pipeline;
    private readonly uint _handle;
    private readonly Dictionary<Value, int> _inputIndex = [];
    private readonly Dictionary<Value, int> _outputIndex = [];
    private readonly byte[]?[] _inputs;
    private readonly int[] _inputLengths;
    private readonly bool[] _dirty;
    private readonly PayloadWriter _writer = new(4096);
    private readonly Queue<Task<WorkerReply>> _earlier = new();
    private readonly IReadOnlyList<string> _producedSymbols;
    private Task<WorkerReply>? _last;
    private RunParameters? _lastParameters;
    private Dictionary<int, ReadOnlyMemory<byte>>? _outputs;
    private int _disposed;

    public RemoteCompiledSegment(RemoteComputeDevice device, WorkerConnection connection, RemotePipeline pipeline, GraphSegment segment, uint handle,
        long deviceMemoryBytes, bool canCopyStateRows, IReadOnlyList<string> producedSymbols)
    {
        _device = device;
        _connection = connection;
        _pipeline = pipeline;
        _handle = handle;
        _producedSymbols = producedSymbols;
        Segment = segment;
        DeviceMemoryBytes = deviceMemoryBytes;
        CanCopyStateRows = canCopyStateRows;
        for (int i = 0; i < segment.Inputs.Count; i++)
        {
            _inputIndex[segment.Inputs[i]] = i;
        }

        for (int i = 0; i < segment.Outputs.Count; i++)
        {
            _outputIndex[segment.Outputs[i]] = i;
        }

        _inputs = new byte[segment.Inputs.Count][];
        _inputLengths = new int[segment.Inputs.Count];
        _dirty = new bool[segment.Inputs.Count];
    }

    public GraphSegment Segment { get; }

    public long DeviceMemoryBytes { get; }

    public bool CanCopyStateRows { get; }

    /// <summary>Asynchronous unless its ops produce data-dependent symbols, which later segments need at once.</summary>
    public bool ExecutesAsynchronously => _producedSymbols.Count == 0;

    /// <summary>How the narrowed boundary values travel.</summary>
    internal BoundaryPrecision Precision { get; init; }

    /// <summary>The inputs that travel in <see cref="Precision"/> (by input index); none by default.</summary>
    internal bool[] NarrowInputs { get; init; } = [];

    /// <summary>The outputs that travel in <see cref="Precision"/> (by output index); none by default.</summary>
    internal bool[] NarrowOutputs { get; init; } = [];

    /// <summary>The worker's handle of the compiled segment.</summary>
    internal uint Handle => _handle;

    /// <summary>The connection to the worker that runs it.</summary>
    internal WorkerConnection Connection => _connection;

    /// <summary>Index of a segment output in the protocol.</summary>
    internal int OutputIndex(Value value) => _outputIndex[value];

    public void SetInput(Value value, ReadOnlySpan<byte> data)
    {
        int i = _inputIndex[value];
        var precision = Narrowed(NarrowInputs, i);
        int length = BoundaryCodec.EncodedLength(data.Length, precision);
        if (_inputs[i] is not { } buffer || buffer.Length < length)
        {
            _inputs[i] = buffer = new byte[length];
        }

        BoundaryCodec.Encode(data, buffer, precision);
        _inputLengths[i] = length;
        _dirty[i] = true;
    }

    public void Execute(RunParameters parameters) => Execute(parameters, s_noSkips);

    public void Execute(RunParameters parameters, IReadOnlySet<Value> skipDownloads)
    {
        ThrowIfFailed();
        CheckEarlier();
        if (_last is not null)
        {
            _earlier.Enqueue(_last);
            _last = null;
        }

        while (_earlier.Count >= MaxOutstanding)
        {
            Complete(_earlier.Dequeue(), parameters: null);
        }

        var w = _writer;
        w.Clear();
        w.UInt32(_handle).Int64((long)_device.Options.ExecuteTimeout.TotalMilliseconds).Map(parameters.Symbols).Map(parameters.Scalars);
        int returned = Segment.Outputs.Count(o => !skipDownloads.Contains(o));
        w.Int32(returned);
        for (int i = 0; i < Segment.Outputs.Count; i++)
        {
            if (!skipDownloads.Contains(Segment.Outputs[i]))
            {
                w.Int32(i);
            }
        }

        w.Int32(_dirty.Count(d => d));
        for (int i = 0; i < _dirty.Length; i++)
        {
            if (_dirty[i])
            {
                w.Int32(i).Bytes(_inputs[i].AsSpan(0, _inputLengths[i]));
                _dirty[i] = false;
            }
        }

        try
        {
            long sent = Stopwatch.GetTimestamp();
            _last = _connection.Request(FrameKind.Execute, w.WrittenSpan);
            _last.ContinueWith(s_record, (_device, sent), CancellationToken.None, TaskContinuationOptions.OnlyOnRanToCompletion | TaskContinuationOptions.ExecuteSynchronously,
                TaskScheduler.Default);
        }
        catch (SegmentWorkerException e)
        {
            throw _pipeline.Fail(e);
        }

        _lastParameters = parameters;
        _outputs = null;
        if (!ExecutesAsynchronously)
        {
            WaitForOutputs();
        }
    }

    public void Prepare(RunParameters parameters, IReadOnlySet<Value> skipDownloads)
    {
        ThrowIfFailed();
        var w = new PayloadWriter().UInt32(_handle).Map(parameters.Symbols);
        w.Int32(skipDownloads.Count);
        foreach (var v in skipDownloads)
        {
            w.Int32(_outputIndex[v]);
        }

        Call(FrameKind.Prepare, w);
    }

    public void GetOutput(Value value, Span<byte> destination)
    {
        WaitForOutputs();
        if (!_outputs!.TryGetValue(_outputIndex[value], out var bytes))
        {
            throw new InvalidOperationException($"Output {value} of remote segment {Segment.Index} was not downloaded in the last execution.");
        }

        var precision = Narrowed(NarrowOutputs, _outputIndex[value]);
        int decoded = BoundaryCodec.DecodedLength(bytes.Length, precision);
        if (decoded < destination.Length)
        {
            throw new InvalidOperationException($"Output {value}: the worker returned {decoded} bytes, {destination.Length} were asked for.");
        }

        BoundaryCodec.Decode(bytes.Span, destination, precision);
    }

    public void ResetState()
    {
        ThrowIfFailed();
        Call(FrameKind.ResetState, new PayloadWriter().UInt32(_handle));
    }

    public void CopyStateRows(IReadOnlyCollection<string> states, int from, int to, int count)
    {
        if (!CanCopyStateRows)
        {
            throw new NotSupportedException("This remote segment cannot copy state rows.");
        }

        ThrowIfFailed();
        Call(FrameKind.CopyStateRows, new PayloadWriter().UInt32(_handle).Strings(states).Int32(from).Int32(to).Int32(count));
    }

    /// <summary>
    /// Takes <paramref name="input"/> straight from <paramref name="producer"/>'s worker when the producer is a remote segment too: the
    /// producer's worker sends it to this segment's worker after every execution (within one worker process it is handed over in memory).
    /// </summary>
    public bool TryForwardFrom(Value input, ICompiledSegment producer)
    {
        if (producer is not RemoteCompiledSegment remote || !_device.Options.PeerForwarding)
        {
            return false;
        }

        int index = _inputIndex[input];
        var reply = Call(FrameKind.AddForwardedInput, new PayloadWriter().UInt32(_handle).Int32(index));
        long tag = new PayloadReader(reply.Payload).Int64();
        string destination = remote.Connection.Info.InstanceId == _connection.Info.InstanceId ? "" : _connection.PeerEndpoint.ToString();
        remote.Call(FrameKind.AddForward, new PayloadWriter().UInt32(remote.Handle).Int32(remote.OutputIndex(input)).String(destination).Int64(tag));
        return true;
    }

    private BoundaryPrecision Narrowed(bool[] flags, int index) => index < flags.Length && flags[index] ? Precision : BoundaryPrecision.Float32;

    /// <summary>Sends a request about this segment and waits for its reply; a failure fails the pipeline.</summary>
    internal Frame Call(FrameKind kind, PayloadWriter payload)
    {
        try
        {
            return _connection.Call(kind, payload.WrittenSpan, _device.Options.ExecuteTimeout, _pipeline.Failed);
        }
        catch (SegmentWorkerException e)
        {
            throw _pipeline.Fail(e);
        }
        catch (OperationCanceledException) when (_pipeline.Failure is not null)
        {
            _pipeline.ThrowIfFailed();
            throw;
        }
    }

    /// <summary>Waits for the reply of the last execution (and the earlier ones) and keeps its outputs.</summary>
    private void WaitForOutputs()
    {
        if (_outputs is not null)
        {
            return;
        }

        ThrowIfFailed();
        while (_earlier.TryDequeue(out var earlier))
        {
            Complete(earlier, parameters: null);
        }

        var last = _last ?? throw new InvalidOperationException("GetOutput called before Execute.");
        _outputs = Complete(last, _lastParameters);
        _last = null;
    }

    /// <summary>Reads an execution's reply: records the timing, applies the symbols its ops produced, returns its outputs.</summary>
    private Dictionary<int, ReadOnlyMemory<byte>> Complete(Task<WorkerReply> reply, RunParameters? parameters)
    {
        WorkerReply arrived;
        try
        {
            arrived = _connection.Wait(reply, _device.Options.ExecuteTimeout, FrameKind.Execute, _pipeline.Failed);
        }
        catch (SegmentWorkerException e)
        {
            throw _pipeline.Fail(e);
        }
        catch (OperationCanceledException) when (_pipeline.Failure is not null)
        {
            _pipeline.ThrowIfFailed();
            throw;
        }

        var r = new PayloadReader(arrived.Frame.Payload);
        r.Double();
        r.Double();
        var symbols = new Dictionary<string, long>(StringComparer.Ordinal);
        r.Map(symbols);
        if (parameters is not null)
        {
            foreach (var (name, value) in symbols)
            {
                parameters.Symbols[name] = value;
            }
        }

        int count = r.Count();
        var outputs = new Dictionary<int, ReadOnlyMemory<byte>>(count);
        for (int i = 0; i < count; i++)
        {
            int index = r.Int32();
            outputs[index] = r.Bytes();
        }

        return outputs;
    }

    /// <summary>Adds an execution's times to the device's statistics when its reply arrives.</summary>
    private static readonly Action<Task<WorkerReply>, object?> s_record = (reply, state) =>
    {
        var (device, sent) = ((RemoteComputeDevice, long))state!;
        var arrived = reply.Result;
        var r = new PayloadReader(arrived.Frame.Payload);
        double workerMs = r.Double();
        double waitMs = r.Double();
        device.Record(Stopwatch.GetElapsedTime(sent, arrived.ArrivedAt).TotalMilliseconds, workerMs, waitMs);
    };

    /// <summary>Surfaces a failure of an execution whose reply nobody waits for (one whose outputs all went to a neighbour).</summary>
    private void CheckEarlier()
    {
        while (_earlier.TryPeek(out var earlier) && earlier.IsCompleted)
        {
            _earlier.Dequeue();
            Complete(earlier, parameters: null);
        }
    }

    private void ThrowIfFailed()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        _pipeline.ThrowIfFailed();
        if (_connection.Fault is { } fault)
        {
            throw _pipeline.Fail(fault);
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _connection.Post(FrameKind.Release, new PayloadWriter().UInt32(_handle).WrittenSpan);
    }
}
