using Sharplight.Distributed.Protocol;
using Sharplight.Distributed.Transport;

namespace Sharplight.Distributed.Worker;

/// <summary>
/// This worker's persistent connections to its neighbours, one per address, over which it sends boundary values
/// (<see cref="FrameKind.PeerValue"/>). A link that fails is dropped and opened again for the next value.
/// </summary>
internal sealed class PeerLinks(ISegmentTransport transport, TimeSpan connectTimeout, string name) : IDisposable
{
    private readonly Dictionary<string, ISegmentConnection> _links = new(StringComparer.Ordinal);
    private readonly Lock _lock = new();
    private bool _disposed;

    /// <summary>Sends a value (or a failure, <paramref name="data"/> empty and <paramref name="error"/> set) to the worker at <paramref name="endpoint"/>.</summary>
    /// <exception cref="IOException">The neighbour cannot be reached or the link broke.</exception>
    public void Send(string endpoint, long tag, long sequence, ReadOnlySpan<byte> data, string? error)
    {
        var link = Get(endpoint);
        var w = new PayloadWriter(data.Length + 32).Int64(tag).Int64(sequence).Bool(error is null);
        if (error is null)
        {
            w.Bytes(data);
        }
        else
        {
            w.String(error);
        }

        try
        {
            link.Send(FrameKind.PeerValue, 0, w.WrittenSpan);
        }
        catch (Exception e) when (e is IOException or ObjectDisposedException)
        {
            Drop(endpoint, link);
            throw new IOException($"Sending a boundary value to neighbour {endpoint} failed: {e.Message}", e);
        }
    }

    private ISegmentConnection Get(string endpoint)
    {
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_links.TryGetValue(endpoint, out var existing))
            {
                return existing;
            }
        }

        var link = Open(SegmentEndpoint.Parse(endpoint));
        lock (_lock)
        {
            if (_disposed || _links.ContainsKey(endpoint))
            {
                link.Dispose();
                ObjectDisposedException.ThrowIf(_disposed, this);
                return _links[endpoint];
            }

            _links[endpoint] = link;
            return link;
        }
    }

    private ISegmentConnection Open(SegmentEndpoint endpoint)
    {
        using var timeout = new CancellationTokenSource(connectTimeout);
        var link = transport.ConnectAsync(endpoint, timeout.Token).GetAwaiter().GetResult();
        try
        {
            var hello = new PayloadWriter().UInt32(SegmentProtocol.Magic).Int32(SegmentProtocol.Version).Byte(SegmentProtocol.PeerRole).String(name);
            link.Send(FrameKind.Hello, 0, hello.WrittenSpan);
            var reply = Task.Run(link.Receive).WaitAsync(connectTimeout).GetAwaiter().GetResult();
            try
            {
                WorkerConnection.ReadHelloAck(reply, endpoint);
            }
            catch (Exception e) when (e is SegmentWorkerException or InvalidDataException)
            {
                throw new IOException($"Neighbour {endpoint}: {e.Message}", e);
            }

            return link;
        }
        catch (TimeoutException e)
        {
            link.Dispose();
            throw new IOException($"Neighbour {endpoint} did not answer the handshake within {connectTimeout.TotalSeconds:F0} s.", e);
        }
        catch
        {
            link.Dispose();
            throw;
        }
    }

    private void Drop(string endpoint, ISegmentConnection link)
    {
        lock (_lock)
        {
            if (_links.TryGetValue(endpoint, out var current) && current == link)
            {
                _links.Remove(endpoint);
            }
        }

        link.Dispose();
    }

    public void Dispose()
    {
        List<ISegmentConnection> links;
        lock (_lock)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            links = [.. _links.Values];
            _links.Clear();
        }

        foreach (var link in links)
        {
            link.Dispose();
        }
    }
}
