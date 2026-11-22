namespace Sharplight.Distributed.Transport;

/// <summary>Accepts the connections of coordinators and neighbouring workers.</summary>
public interface ISegmentListener : IDisposable
{
    /// <summary>The bound address (with the port the system chose when 0 was asked for).</summary>
    public SegmentEndpoint LocalEndpoint { get; }

    /// <summary>Waits for the next connection (after its transport handshake, e.g. TLS).</summary>
    /// <exception cref="OperationCanceledException">Cancelled, or the listener was disposed.</exception>
    public Task<ISegmentConnection> AcceptAsync(CancellationToken cancellationToken);
}
