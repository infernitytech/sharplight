using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using Sharplight.Distributed.Protocol;
using Sharplight.Distributed.Transport;
using Sharplight.Package;
using Sharplight.Runtime;

namespace Sharplight.Distributed.Worker;

/// <summary>
/// One coordinator's connection to a worker: the segments it compiled here and their executions. A reader thread answers heartbeats, memory
/// and support questions and aborts at once; compilations, executions and state operations run in order on an executor thread. When the
/// connection closes or the coordinator goes silent, everything compiled over it is released.
/// </summary>
internal sealed class WorkerSession : IDisposable
{
    private readonly SegmentWorker _worker;
    private readonly ISegmentConnection _connection;
    private readonly Dictionary<uint, HostedSegment> _segments = [];
    private readonly BlockingCollection<(Frame Frame, long Index, long Received)> _queue = new();
    private readonly CancellationTokenSource _closed = new();
    private readonly Timer _watchdog;
    private CancellationTokenSource _abort = new();
    private long _enqueued;
    private long _abortedThrough;
    private long _lastReceived = Stopwatch.GetTimestamp();
    private readonly PayloadWriter _reply = new(64 << 10);
    private byte[] _scratch = [];
    private uint _nextHandle;
    /// <summary>When the request the executor handles arrived (the time it queued behind earlier work counts as waiting).</summary>
    private long _received;
    private int _disposed;

    public WorkerSession(SegmentWorker worker, ISegmentConnection connection)
    {
        _worker = worker;
        _connection = connection;
        var period = TimeSpan.FromSeconds(Math.Max(0.25, worker.Options.HeartbeatTimeout.TotalSeconds / 4));
        _watchdog = new Timer(_ => Watch(), null, period, period);
    }

    /// <summary>Serves the connection until it closes (reads on the calling thread, executes on a thread of its own).</summary>
    public void Run()
    {
        var executor = new Thread(Execute) { IsBackground = true, Name = $"sharplight-worker {_connection.RemoteName}" };
        executor.Start();
        try
        {
            while (true)
            {
                var frame = _connection.Receive();
                Volatile.Write(ref _lastReceived, Stopwatch.GetTimestamp());
                switch (frame.Kind)
                {
                    case FrameKind.Ping:
                        Send(FrameKind.Pong, frame.RequestId, []);
                        break;
                    case FrameKind.Abort:
                        Abort();
                        break;
                    case FrameKind.MemoryRequest:
                        Answer(frame, _ => Memory());
                        break;
                    case FrameKind.SupportsRequest:
                        Answer(frame, Supports);
                        break;
                    default:
                        _queue.Add((frame, Interlocked.Increment(ref _enqueued), Stopwatch.GetTimestamp()));
                        break;
                }
            }
        }
        catch (Exception e) when (e is IOException or InvalidOperationException or ObjectDisposedException)
        {
            if (Volatile.Read(ref _disposed) == 0)
            {
                _worker.Log($"coordinator {_connection.RemoteName} disconnected: {e.Message}");
            }
        }
        finally
        {
            Dispose();
            executor.Join();
            ReleaseAll();
        }
    }

    /// <summary>Fails the execution in progress and every request queued before the abort (the coordinator's pipeline failed).</summary>
    private void Abort()
    {
        Volatile.Write(ref _abortedThrough, Interlocked.Read(ref _enqueued));
        Interlocked.Exchange(ref _abort, new CancellationTokenSource()).Cancel();
    }

    private void Watch()
    {
        var silent = Stopwatch.GetElapsedTime(Volatile.Read(ref _lastReceived));
        if (silent > _worker.Options.HeartbeatTimeout)
        {
            _worker.Log($"coordinator {_connection.RemoteName} silent for {silent.TotalSeconds:F1} s; releasing its segments");
            Dispose();
        }
    }

    private void Execute()
    {
        foreach (var (frame, index, received) in _queue.GetConsumingEnumerable())
        {
            _received = received;
            if (frame.Kind == FrameKind.Release)
            {
                Release(new PayloadReader(frame.Payload).UInt32());
                continue;
            }

            if (index <= Volatile.Read(ref _abortedThrough))
            {
                Fail(frame.RequestId, new OperationCanceledException("The coordinator aborted the execution."));
                continue;
            }

            Answer(frame, frame.Kind switch
            {
                FrameKind.Compile => Compile,
                FrameKind.Execute => Run,
                FrameKind.Prepare => Prepare,
                FrameKind.ResetState => ResetState,
                FrameKind.CopyStateRows => CopyStateRows,
                FrameKind.AddForward => AddForward,
                FrameKind.AddForwardedInput => AddForwardedInput,
                _ => static _ => throw new InvalidDataException("Unknown request."),
            });
        }
    }

    /// <summary>Runs a request handler and sends its reply, or the error.</summary>
    private void Answer(Frame frame, Func<PayloadReader, (FrameKind Kind, PayloadWriter? Payload)> handler)
    {
        try
        {
            var (kind, payload) = handler(new PayloadReader(frame.Payload));
            Send(kind, frame.RequestId, payload is null ? [] : payload.WrittenSpan);
        }
        catch (Exception e)
        {
            Fail(frame.RequestId, e);
        }
    }

    private void Fail(uint requestId, Exception e)
    {
        byte kind = e is OutOfMemoryException or InsufficientMemoryException ? SegmentProtocol.OutOfMemoryError
            : e is ModelMismatchException ? SegmentProtocol.ModelMismatchError
            : SegmentProtocol.GeneralError;
        string message = e is OperationCanceledException ? "aborted by the coordinator" : $"{e.GetType().Name}: {e.Message}";
        if (e is not OperationCanceledException)
        {
            _worker.Log($"request from {_connection.RemoteName} failed: {message}");
        }

        Send(FrameKind.Error, requestId, new PayloadWriter().Byte(kind).String(message).WrittenSpan);
    }

    private (FrameKind, PayloadWriter?) Memory()
    {
        var devices = _worker.Describe();
        var w = new PayloadWriter().Int32(devices.Count);
        foreach (var d in devices)
        {
            w.Int64(d.TotalMemory).Int64(d.AvailableMemory);
        }

        return (FrameKind.MemoryReply, w);
    }

    private (FrameKind, PayloadWriter?) Supports(PayloadReader r)
    {
        var graph = ReadGraph(r.Rest(), out _);
        var node = graph.Nodes[^1];
        var options = _worker.Options;
        var unsupported = options.Devices.Select(d => d.Device.Supports(node)).FirstOrDefault(s => !s.IsSupported);
        bool fallback = options.Fallback is { } f && f.Device.Supports(node).IsSupported;
        var w = new PayloadWriter().Bool(unsupported.Reason is null || fallback);
        w.String(unsupported.Reason ?? "");
        return (FrameKind.SupportsReply, w);
    }

    private (FrameKind, PayloadWriter?) Compile(PayloadReader r)
    {
        var stopwatch = Stopwatch.StartNew();
        var options = CompileOptionsCodec.Read(r, _worker.SharedWeights, _worker.DiskExperts, _closed.Token);
        var verification = (WeightVerification)r.Byte();
        string fingerprint = r.String();
        var graph = ReadGraph(r.Bytes(), out var outputNames);
        var precision = (BoundaryPrecision)r.Byte();
        if (!Enum.IsDefined(precision))
        {
            throw new InvalidDataException($"Unknown boundary precision {(int)precision}.");
        }

        var narrowInputs = r.Indices(graph.Inputs.Count);
        var narrowOutputs = r.Indices(outputNames.Count);
        string local = SegmentFingerprint.Compute(graph.Weights.Select(v => v.Weight!), verification);
        if (local != fingerprint)
        {
            throw new ModelMismatchException($"The weights of segment {graph.Name} differ from the coordinator's ({verification} check): this worker's model copy is another model, quantization or revision.");
        }

        var worker = _worker.Options;
        var placement = new PlacementOptions
        {
            Devices = [.. worker.Devices],
            Fallback = worker.Fallback,
            ReservePerDevice = worker.ReservePerDevice ?? new PlacementOptions { Devices = [], Compile = options }.ReservePerDevice,
            Compile = options,
        };
        var model = CompiledModel.Compile(graph, placement);
        uint handle = ++_nextHandle;
        lock (_segments)
        {
            _segments[handle] = new HostedSegment(model, graph, outputNames)
            {
                Precision = precision,
                NarrowInputs = narrowInputs,
                NarrowOutputs = narrowOutputs,
            };
        }

        _worker.Log($"compiled {graph.Name} ({graph.Nodes.Count} nodes, {model.ResidentMemoryBytes / (1024.0 * 1024 * 1024):F2} GiB) for {_connection.RemoteName} "
            + $"in {stopwatch.ElapsedMilliseconds} ms\n{model.Plan.Describe()}");
        return (FrameKind.CompileReply, new PayloadWriter().UInt32(handle).Int64(model.ResidentMemoryBytes).Bool(model.CanCopyStateRows));
    }

    private (FrameKind, PayloadWriter?) Run(PayloadReader r)
    {
        var segment = Segment(r.UInt32());
        var timeout = TimeSpan.FromMilliseconds(r.Int64());
        var abort = Volatile.Read(ref _abort).Token;
        var p = segment.Parameters;
        p.Symbols.Clear();
        p.Scalars.Clear();
        p.SkipOutputs.Clear();
        r.Map(p.Symbols);
        r.Map(p.Scalars);
        int returnedCount = r.Count();
        var returned = new int[returnedCount];
        for (int i = 0; i < returnedCount; i++)
        {
            returned[i] = r.Int32();
        }

        long sequence = segment.Executions++;
        int inputs = r.Count();
        for (int i = 0; i < inputs; i++)
        {
            int index = r.Int32();
            SetInput(segment, index, r.Bytes().Span);
        }

        try
        {
            foreach (var (input, tag) in segment.ForwardedInputs)
            {
                var data = _worker.Mailbox.Take(tag, sequence, timeout, abort);
                SetInput(segment, input, data);
            }

            var started = Stopwatch.GetTimestamp();
            double waitMs = Stopwatch.GetElapsedTime(_received, started).TotalMilliseconds;
            var needed = returned.Concat(segment.Forwards.Select(f => f.Output)).ToHashSet();
            for (int i = 0; i < segment.OutputNames.Count; i++)
            {
                if (!needed.Contains(i))
                {
                    p.SkipOutputs.Add(segment.OutputNames[i]);
                }
            }

            segment.Model.Run(p);
            byte[]? forwarded = null;
            int forwardedOutput = -1;
            foreach (var (output, endpoint, tag) in segment.Forwards)
            {
                if (output != forwardedOutput)
                {
                    forwarded = new byte[WireBytes(segment, output)];
                    ReadOutput(segment, output, forwarded);
                    forwardedOutput = output;
                }

                if (endpoint.Length == 0)
                {
                    _worker.Mailbox.Post(tag, sequence, forwarded, null, source: null);
                }
                else
                {
                    _worker.Links.Send(endpoint, tag, sequence, forwarded!, null);
                }
            }

            // The reply is built in place: the outputs are read straight into it.
            var w = _reply;
            w.Clear();
            int workerMsAt = w.Length;
            w.Double(0).Double(waitMs).Map(ProducedSymbols(segment, p)).Int32(returned.Length);
            foreach (int output in returned)
            {
                w.Int32(output);
                ReadOutput(segment, output, w.Block(WireBytes(segment, output)));
            }

            w.SetDouble(workerMsAt, Stopwatch.GetElapsedTime(started).TotalMilliseconds);
            return (FrameKind.ExecuteReply, w);
        }
        catch (Exception e)
        {
            // The neighbours that wait for this segment's outputs fail at once instead of at their timeout.
            foreach (var (_, endpoint, tag) in segment.Forwards)
            {
                try
                {
                    if (endpoint.Length == 0)
                    {
                        _worker.Mailbox.Post(tag, sequence, null, e.Message, source: null);
                    }
                    else
                    {
                        _worker.Links.Send(endpoint, tag, sequence, [], e.Message);
                    }
                }
                catch (IOException)
                {
                    // The neighbour is gone too; its coordinator learns it from its own connection.
                }
            }

            throw;
        }
    }

    private (FrameKind, PayloadWriter?) Prepare(PayloadReader r)
    {
        var segment = Segment(r.UInt32());
        var p = new RunParameters();
        r.Map(p.Symbols);
        int skipped = r.Count();
        for (int i = 0; i < skipped; i++)
        {
            p.SkipOutputs.Add(segment.OutputNames[r.Int32()]);
        }

        segment.Model.Prepare(p);
        return (FrameKind.Ack, null);
    }

    private (FrameKind, PayloadWriter?) ResetState(PayloadReader r)
    {
        Segment(r.UInt32()).Model.ResetState();
        return (FrameKind.Ack, null);
    }

    private (FrameKind, PayloadWriter?) CopyStateRows(PayloadReader r)
    {
        var segment = Segment(r.UInt32());
        var states = r.Strings();
        int from = r.Int32();
        int to = r.Int32();
        int count = r.Int32();
        if (!segment.Model.CopyStateRows(states, from, to, count))
        {
            throw new NotSupportedException("The segment cannot copy state rows.");
        }

        return (FrameKind.Ack, null);
    }

    private (FrameKind, PayloadWriter?) AddForward(PayloadReader r)
    {
        var segment = Segment(r.UInt32());
        int output = r.Int32();
        string endpoint = r.String();
        long tag = r.Int64();
        segment.Forwards.Add((output, endpoint, tag));
        return (FrameKind.Ack, null);
    }

    private (FrameKind, PayloadWriter?) AddForwardedInput(PayloadReader r)
    {
        var segment = Segment(r.UInt32());
        int input = r.Int32();
        long tag = _worker.Mailbox.Register();
        segment.ForwardedInputs.Add((input, tag));
        return (FrameKind.ForwardedInputReply, new PayloadWriter().Int64(tag));
    }

    private HostedSegment Segment(uint handle)
    {
        lock (_segments)
        {
            return _segments.TryGetValue(handle, out var segment) ? segment : throw new InvalidOperationException($"No compiled segment {handle} on this connection.");
        }
    }

    private void Release(uint handle)
    {
        HostedSegment? segment;
        lock (_segments)
        {
            _segments.Remove(handle, out segment);
        }

        if (segment is not null)
        {
            Unregister(segment);
            segment.Dispose();
        }
    }

    private void ReleaseAll()
    {
        List<HostedSegment> segments;
        lock (_segments)
        {
            segments = [.. _segments.Values];
            _segments.Clear();
        }

        foreach (var segment in segments)
        {
            Unregister(segment);
            segment.Dispose();
        }

        if (segments.Count > 0)
        {
            _worker.Log($"released {segments.Count} segment(s) of {_connection.RemoteName}");
        }
    }

    private void Unregister(HostedSegment segment)
    {
        foreach (var (_, tag) in segment.ForwardedInputs)
        {
            _worker.Mailbox.Unregister(tag);
        }
    }

    private Sharplight.Graph.ComputeGraph ReadGraph(ReadOnlyMemory<byte> blob, out IReadOnlyList<string> outputNames)
    {
        using var stream = new MemoryStream(blob.ToArray(), writable: false);
        using var reader = new BinaryReader(stream, Encoding.UTF8);
        try
        {
            return SegmentCodec.Read(reader, _worker.Weight, out outputNames);
        }
        catch (InvalidDataException e)
        {
            throw new ModelMismatchException(e.Message, e);
        }
    }

    /// <summary>Bytes of output <paramref name="output"/> on the wire in the current execution.</summary>
    private static int WireBytes(HostedSegment segment, int output) => BoundaryCodec.EncodedLength(segment.OutputBytes(output), segment.OutputPrecision(output));

    /// <summary>Reads output <paramref name="output"/> into <paramref name="destination"/> as it goes on the wire (<see cref="WireBytes"/>).</summary>
    private void ReadOutput(HostedSegment segment, int output, Span<byte> destination)
    {
        var precision = segment.OutputPrecision(output);
        if (precision == BoundaryPrecision.Float32)
        {
            segment.Model.GetOutput(segment.OutputNames[output], destination);
            return;
        }

        var values = Scratch(segment.OutputBytes(output));
        segment.Model.GetOutput(segment.OutputNames[output], values);
        BoundaryCodec.Encode(values, destination, precision);
    }

    /// <summary>Sets input <paramref name="index"/> from its bytes on the wire.</summary>
    private void SetInput(HostedSegment segment, int index, ReadOnlySpan<byte> data)
    {
        var precision = segment.InputPrecision(index);
        if (precision == BoundaryPrecision.Float32)
        {
            segment.Model.SetInput(segment.Graph.Inputs[index], data);
            return;
        }

        var values = Scratch(BoundaryCodec.DecodedLength(data.Length, precision));
        BoundaryCodec.Decode(data, values, precision);
        segment.Model.SetInput(segment.Graph.Inputs[index], values);
    }

    /// <summary>The executor's buffer for widened values (valid until the next call).</summary>
    private Span<byte> Scratch(int length)
    {
        if (_scratch.Length < length)
        {
            _scratch = new byte[Math.Max(length, _scratch.Length * 2)];
        }

        return _scratch.AsSpan(0, length);
    }

    /// <summary>The data-dependent symbols the execution produced.</summary>
    private static Dictionary<string, long> ProducedSymbols(HostedSegment segment, RunParameters p)
    {
        var produced = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var (symbol, _) in segment.Graph.DynamicSymbols)
        {
            if (p.Symbols.TryGetValue(symbol, out long value))
            {
                produced[symbol] = value;
            }
        }

        return produced;
    }

    private void Send(FrameKind kind, uint requestId, ReadOnlySpan<byte> payload)
    {
        try
        {
            _connection.Send(kind, requestId, payload);
        }
        catch (Exception e) when (e is IOException or ObjectDisposedException)
        {
            Dispose();
        }
    }

    /// <summary>Closes the connection (the reader then releases the segments).</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _watchdog.Dispose();
        _closed.Cancel();
        _abort.Cancel();
        _queue.CompleteAdding();
        _connection.Dispose();
        _worker.Remove(this);
    }
}
