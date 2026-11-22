using System.Collections.Concurrent;
using System.Diagnostics;
using Sharplight.Distributed.Protocol;
using Sharplight.Distributed.Transport;

namespace Sharplight.Distributed;

/// <summary>
/// The coordinator's persistent connection to one segment worker: requests with replies matched by id, a reader thread, heartbeats and
/// the failure semantics. Once the connection fails (the worker closed it, stopped answering heartbeats, or a request timed out) every
/// pending and later request throws <see cref="SegmentWorkerException"/>; the connection is never reused.
/// </summary>
internal sealed class WorkerConnection : IDisposable
{
    private readonly ISegmentConnection _connection;
    private readonly RemoteSegmentOptions _options;
    private readonly ConcurrentDictionary<uint, TaskCompletionSource<WorkerReply>> _pending = new();
    private readonly Thread _reader;
    private readonly Timer _heartbeat;
    private long _lastReceived;
    private long _lastSent;
    private uint _nextRequest;
    private SegmentWorkerException? _fault;
    private int _disposed;
    private long _bytesSent;
    private long _bytesReceived;

    private WorkerConnection(ISegmentConnection connection, SegmentEndpoint endpoint, SegmentWorkerInfo info, RemoteSegmentOptions options)
    {
        _connection = connection;
        _options = options;
        Endpoint = endpoint;
        Info = info;
        _lastReceived = _lastSent = Stopwatch.GetTimestamp();
        _reader = new Thread(ReadLoop) { IsBackground = true, Name = $"sharplight-remote {endpoint}" };
        _reader.Start();
        var period = options.HeartbeatInterval > TimeSpan.Zero ? options.HeartbeatInterval : TimeSpan.FromSeconds(1);
        _heartbeat = new Timer(_ => Heartbeat(), null, period, period);
    }

    /// <summary>The address the coordinator connected to.</summary>
    public SegmentEndpoint Endpoint { get; }

    /// <summary>What the worker reported in its handshake.</summary>
    public SegmentWorkerInfo Info { get; }

    /// <summary>Where neighbouring workers reach this worker.</summary>
    public SegmentEndpoint PeerEndpoint => Info.AdvertisedEndpoint ?? Endpoint;

    /// <summary>Why the connection failed, or null while it works.</summary>
    public SegmentWorkerException? Fault => Volatile.Read(ref _fault);

    /// <summary>Raised once when the connection fails while in use (not when it is disposed), on the thread that noticed the failure.</summary>
    public event Action<SegmentWorkerException>? Faulted;

    /// <summary>Bytes of frames sent and received (payloads plus headers).</summary>
    public long BytesSent => Interlocked.Read(ref _bytesSent);

    /// <inheritdoc cref="BytesSent"/>
    public long BytesReceived => Interlocked.Read(ref _bytesReceived);

    /// <summary>Connects to a worker and runs the handshake.</summary>
    /// <exception cref="SegmentWorkerException">The worker cannot be reached, speaks another protocol version or refused the connection.</exception>
    public static async Task<WorkerConnection> OpenAsync(ISegmentTransport transport, SegmentEndpoint endpoint, RemoteSegmentOptions options, CancellationToken cancellationToken)
    {
        ISegmentConnection connection;
        try
        {
            connection = await transport.ConnectAsync(endpoint, cancellationToken).ConfigureAwait(false);
        }
        catch (IOException e)
        {
            throw new SegmentWorkerException(e.Message, e) { Worker = endpoint.ToString() };
        }

        try
        {
            var hello = new PayloadWriter().UInt32(SegmentProtocol.Magic).Int32(SegmentProtocol.Version).Byte(SegmentProtocol.CoordinatorRole).String(Environment.MachineName);
            connection.Send(FrameKind.Hello, 0, hello.WrittenSpan);
            var reply = await Task.Run(connection.Receive, cancellationToken).WaitAsync(options.HeartbeatTimeout + TimeSpan.FromSeconds(5), cancellationToken).ConfigureAwait(false);
            var info = ReadHelloAck(reply, endpoint);
            return new WorkerConnection(connection, endpoint, info, options);
        }
        catch (Exception e) when (e is IOException or TimeoutException or InvalidDataException)
        {
            connection.Dispose();
            throw new SegmentWorkerException($"Segment worker {endpoint}: handshake failed: {e.Message}", e) { Worker = endpoint.ToString() };
        }
        catch
        {
            connection.Dispose();
            throw;
        }
    }

    /// <summary>Parses the worker's answer to the hello.</summary>
    /// <exception cref="SegmentWorkerException">The worker refused the connection.</exception>
    /// <exception cref="InvalidDataException">The answer is not a hello answer of this protocol version.</exception>
    internal static SegmentWorkerInfo ReadHelloAck(Frame reply, SegmentEndpoint endpoint)
    {
        var r = new PayloadReader(reply.Payload);
        if (reply.Kind == FrameKind.Error)
        {
            r.Byte();
            throw new SegmentWorkerException($"Segment worker {endpoint} refused the connection: {r.String()}") { Worker = endpoint.ToString() };
        }

        if (reply.Kind != FrameKind.HelloAck || r.UInt32() != SegmentProtocol.Magic)
        {
            throw new InvalidDataException($"{endpoint} is not a Sharplight segment worker.");
        }

        int version = r.Int32();
        if (version != SegmentProtocol.Version)
        {
            throw new InvalidDataException($"protocol version {version}, this coordinator speaks {SegmentProtocol.Version}.");
        }

        return SegmentWorkerInfo.Read(r);
    }

    /// <summary>Sends a request; the task completes with its reply, or fails with <see cref="SegmentWorkerException"/>.</summary>
    public Task<WorkerReply> Request(FrameKind kind, ReadOnlySpan<byte> payload)
    {
        ThrowIfFaulted();
        uint id = Interlocked.Increment(ref _nextRequest);
        var reply = new TaskCompletionSource<WorkerReply>(TaskCreationOptions.RunContinuationsAsynchronously);
        _pending[id] = reply;
        try
        {
            Send(kind, id, payload);
        }
        catch (SegmentWorkerException)
        {
            _pending.TryRemove(id, out _);
            throw;
        }

        if (Fault is { } fault && _pending.TryRemove(id, out _))
        {
            reply.TrySetException(fault);
        }

        return reply.Task;
    }

    /// <summary>Sends a request and waits for its reply.</summary>
    public Frame Call(FrameKind kind, ReadOnlySpan<byte> payload, TimeSpan timeout, CancellationToken cancellationToken = default)
        => Wait(Request(kind, payload), timeout, kind, cancellationToken).Frame;

    /// <summary>
    /// Waits for a reply. A reply that does not come within <paramref name="timeout"/> fails the connection: the worker's state is unknown
    /// afterwards.
    /// </summary>
    /// <exception cref="SegmentWorkerException">The request failed, the connection failed, or the timeout passed.</exception>
    /// <exception cref="OperationCanceledException">Cancelled (the request still runs on the worker; the connection stays usable).</exception>
    public WorkerReply Wait(Task<WorkerReply> reply, TimeSpan timeout, FrameKind kind, CancellationToken cancellationToken = default)
    {
        try
        {
            if (!reply.Wait(timeout, cancellationToken))
            {
                var fault = Fail($"Segment worker {Endpoint} did not answer a {kind} request within {timeout.TotalSeconds:F0} s.", null);
                throw fault;
            }
        }
        catch (AggregateException e) when (e.InnerException is SegmentWorkerException inner)
        {
            throw new SegmentWorkerException(inner.Message, inner) { Worker = inner.Worker, WorkerAlive = inner.WorkerAlive };
        }

        return reply.Result;
    }

    /// <summary>Sends a frame that has no reply (best effort: nothing happens when the connection already failed).</summary>
    public void Post(FrameKind kind, ReadOnlySpan<byte> payload)
    {
        if (Fault is not null || Volatile.Read(ref _disposed) != 0)
        {
            return;
        }

        try
        {
            Send(kind, 0, payload);
        }
        catch (SegmentWorkerException)
        {
            // The failure is recorded; a later request reports it.
        }
    }

    /// <exception cref="SegmentWorkerException">The connection failed.</exception>
    public void ThrowIfFaulted()
    {
        if (Fault is { } fault)
        {
            throw new SegmentWorkerException(fault.Message, fault) { Worker = fault.Worker };
        }

        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
    }

    private void Send(FrameKind kind, uint id, ReadOnlySpan<byte> payload)
    {
        try
        {
            _connection.Send(kind, id, payload);
            Interlocked.Add(ref _bytesSent, payload.Length + FrameHeader.Size);
            Volatile.Write(ref _lastSent, Stopwatch.GetTimestamp());
        }
        catch (Exception e) when (e is IOException or ObjectDisposedException)
        {
            throw Fail($"Sending to segment worker {Endpoint} failed: {e.Message}", e);
        }
    }

    private void ReadLoop()
    {
        try
        {
            while (true)
            {
                var frame = _connection.Receive();
                Interlocked.Add(ref _bytesReceived, frame.Payload.Length + FrameHeader.Size);
                Volatile.Write(ref _lastReceived, Stopwatch.GetTimestamp());
                if (frame.Kind == FrameKind.Pong || frame.RequestId == 0)
                {
                    continue;
                }

                if (!_pending.TryRemove(frame.RequestId, out var reply))
                {
                    continue;
                }

                if (frame.Kind == FrameKind.Error)
                {
                    var r = new PayloadReader(frame.Payload);
                    byte errorKind = r.Byte();
                    string message = r.String();
                    reply.TrySetException(new SegmentWorkerException(errorKind == SegmentProtocol.OutOfMemoryError
                        ? $"Segment worker {Endpoint} ran out of memory: {message}"
                        : $"Segment worker {Endpoint}: {message}") { Worker = Endpoint.ToString(), WorkerAlive = true });
                }
                else
                {
                    reply.TrySetResult(new WorkerReply(frame, Stopwatch.GetTimestamp()));
                }
            }
        }
        catch (Exception e)
        {
            Fail(Volatile.Read(ref _disposed) != 0 ? $"The connection to segment worker {Endpoint} was closed." : $"Lost segment worker {Endpoint}: {e.Message}", e);
        }
    }

    private void Heartbeat()
    {
        if (Fault is not null || Volatile.Read(ref _disposed) != 0)
        {
            return;
        }

        var silent = Stopwatch.GetElapsedTime(Volatile.Read(ref _lastReceived));
        if (silent > _options.HeartbeatTimeout)
        {
            Fail($"Segment worker {Endpoint} sent nothing for {silent.TotalSeconds:F1} s (heartbeat timeout {_options.HeartbeatTimeout.TotalSeconds:F1} s).", null);
            return;
        }

        if (Stopwatch.GetElapsedTime(Volatile.Read(ref _lastSent)) >= _options.HeartbeatInterval / 2)
        {
            Post(FrameKind.Ping, []);
        }
    }

    /// <summary>Marks the connection failed (the first reason wins), fails every pending request and closes the connection.</summary>
    private SegmentWorkerException Fail(string message, Exception? cause)
    {
        var fault = cause is null
            ? new SegmentWorkerException(message) { Worker = Endpoint.ToString() }
            : new SegmentWorkerException(message, cause) { Worker = Endpoint.ToString() };
        var earlier = Interlocked.CompareExchange(ref _fault, fault, null);
        var first = earlier ?? fault;
        foreach (var id in _pending.Keys)
        {
            if (_pending.TryRemove(id, out var reply))
            {
                reply.TrySetException(first);
            }
        }

        _connection.Dispose();
        if (earlier is null && Volatile.Read(ref _disposed) == 0)
        {
            Faulted?.Invoke(first);
        }

        return first;
    }

    /// <summary>Closes the connection; the worker releases everything compiled over it.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _heartbeat.Dispose();
        Fail($"The connection to segment worker {Endpoint} was closed.", null);
        if (Thread.CurrentThread != _reader)
        {
            _reader.Join(TimeSpan.FromSeconds(5));
        }
    }
}
