using System.Net.Sockets;

namespace Sharplight.Distributed.Transport;

/// <summary>A listening TCP socket of <see cref="TcpSegmentTransport"/>.</summary>
internal sealed class TcpSegmentListener(TcpSegmentTransport transport, Socket socket, SegmentEndpoint endpoint) : ISegmentListener
{
    public SegmentEndpoint LocalEndpoint => endpoint;

    public async Task<ISegmentConnection> AcceptAsync(CancellationToken cancellationToken)
    {
        Socket accepted;
        try
        {
            accepted = await socket.AcceptAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (Exception e) when (e is ObjectDisposedException or SocketException { SocketErrorCode: SocketError.OperationAborted or SocketError.Interrupted })
        {
            throw new OperationCanceledException("The listener was closed.", e, cancellationToken);
        }

        return await transport.AcceptedAsync(accepted, cancellationToken).ConfigureAwait(false);
    }

    public void Dispose() => socket.Dispose();
}
