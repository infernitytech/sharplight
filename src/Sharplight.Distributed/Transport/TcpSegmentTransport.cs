using System.Net;
using System.Net.Security;
using System.Net.Sockets;

namespace Sharplight.Distributed.Transport;

/// <summary>
/// Segment frames over TCP: one persistent connection per peer, <c>TCP_NODELAY</c> (a decode step's activations are small and must
/// not wait for Nagle's algorithm), keep-alive, large socket buffers, and optional TLS or mutual TLS (<see cref="TcpSegmentTransportOptions"/>).
/// </summary>
public sealed class TcpSegmentTransport : ISegmentTransport
{
    private readonly TcpSegmentTransportOptions _options;

    /// <summary>Creates the transport.</summary>
    /// <param name="options">Timeouts, TLS and limits, or null for the defaults (no TLS).</param>
    public TcpSegmentTransport(TcpSegmentTransportOptions? options = null) => _options = options ?? new TcpSegmentTransportOptions();

    /// <inheritdoc/>
    public string Name => "tcp";

    /// <inheritdoc/>
    public async Task<ISegmentConnection> ConnectAsync(SegmentEndpoint endpoint, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(endpoint);
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(_options.ConnectTimeout);
        var socket = new Socket(SocketType.Stream, ProtocolType.Tcp);
        try
        {
            Configure(socket);
            await socket.ConnectAsync(endpoint.Host, endpoint.Port, timeout.Token).ConfigureAwait(false);
            return await WrapAsync(socket, endpoint.ToString(), client: endpoint, timeout.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            socket.Dispose();
            throw new IOException($"Connecting to segment worker {endpoint} timed out after {_options.ConnectTimeout.TotalSeconds:F0} s.");
        }
        catch (Exception e) when (e is SocketException or System.Security.Authentication.AuthenticationException)
        {
            socket.Dispose();
            throw new IOException($"Cannot connect to segment worker {endpoint}: {e.Message}", e);
        }
        catch
        {
            socket.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public ISegmentListener Listen(SegmentEndpoint endpoint)
    {
        ArgumentNullException.ThrowIfNull(endpoint);
        var address = endpoint.Host is "*" or "0.0.0.0" ? IPAddress.Any
            : endpoint.Host is "::" ? IPAddress.IPv6Any
            : IPAddress.TryParse(endpoint.Host, out var parsed) ? parsed
            : Dns.GetHostAddresses(endpoint.Host).First();
        var socket = new Socket(address.AddressFamily, SocketType.Stream, ProtocolType.Tcp);
        try
        {
            socket.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
            socket.Bind(new IPEndPoint(address, endpoint.Port));
            socket.Listen(64);
        }
        catch
        {
            socket.Dispose();
            throw;
        }

        var bound = (IPEndPoint)socket.LocalEndPoint!;
        return new TcpSegmentListener(this, socket, new SegmentEndpoint(endpoint.Host, bound.Port));
    }

    /// <summary>Sets up an accepted socket and runs the server side of the TLS handshake.</summary>
    internal async Task<ISegmentConnection> AcceptedAsync(Socket socket, CancellationToken cancellationToken)
    {
        try
        {
            Configure(socket);
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(_options.ConnectTimeout);
            return await WrapAsync(socket, socket.RemoteEndPoint?.ToString() ?? "peer", client: null, timeout.Token).ConfigureAwait(false);
        }
        catch
        {
            socket.Dispose();
            throw;
        }
    }

    private void Configure(Socket socket)
    {
        socket.NoDelay = true;
        socket.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.KeepAlive, true);
        if (_options.SocketBufferBytes > 0)
        {
            socket.SendBufferSize = _options.SocketBufferBytes;
            socket.ReceiveBufferSize = _options.SocketBufferBytes;
        }
    }

    private async Task<ISegmentConnection> WrapAsync(Socket socket, string name, SegmentEndpoint? client, CancellationToken cancellationToken)
    {
        var network = new NetworkStream(socket, ownsSocket: true);
        SslStream? tls = null;
        try
        {
            if (client is not null && _options.ClientTls?.Invoke(client) is { } clientOptions)
            {
                clientOptions.TargetHost ??= client.Host;
                tls = new SslStream(network, leaveInnerStreamOpen: false);
                await tls.AuthenticateAsClientAsync(clientOptions, cancellationToken).ConfigureAwait(false);
            }
            else if (client is null && _options.ServerTls is { } serverOptions)
            {
                tls = new SslStream(network, leaveInnerStreamOpen: false);
                await tls.AuthenticateAsServerAsync(serverOptions, cancellationToken).ConfigureAwait(false);
            }
        }
        catch
        {
            tls?.Dispose();
            network.Dispose();
            throw;
        }

        return new StreamSegmentConnection(tls ?? (Stream)network, owner: null, name, _options.MaxFrameBytes);
    }
}
