namespace Sharplight.Distributed.Transport;

/// <summary>
/// How segment frames travel between processes. <see cref="TcpSegmentTransport"/> is the implementation for Ethernet; a transport for
/// RDMA networks (InfiniBand, RoCE) implements the same contract, e.g. with registered buffers for large payloads.
/// </summary>
public interface ISegmentTransport
{
    /// <summary>Short name for messages (<c>tcp</c>).</summary>
    public string Name { get; }

    /// <summary>Opens a connection to a listening worker.</summary>
    /// <exception cref="IOException">The worker cannot be reached in time, or the handshake failed.</exception>
    public Task<ISegmentConnection> ConnectAsync(SegmentEndpoint endpoint, CancellationToken cancellationToken);

    /// <summary>Starts listening on an address (port 0: a free port).</summary>
    public ISegmentListener Listen(SegmentEndpoint endpoint);
}
