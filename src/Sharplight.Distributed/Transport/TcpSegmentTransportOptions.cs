using System.Net.Security;

namespace Sharplight.Distributed.Transport;

/// <summary>Settings of <see cref="TcpSegmentTransport"/>.</summary>
public sealed class TcpSegmentTransportOptions
{
    /// <summary>How long a connection attempt (TCP connect plus TLS handshake) may take. Default 10 s.</summary>
    public TimeSpan ConnectTimeout { get; set; } = TimeSpan.FromSeconds(10);

    /// <summary>
    /// TLS for outgoing connections: returns the client options for a worker (certificate for mutual TLS, validation callback, target host),
    /// or null for a plain connection. Null (the default): no TLS. When the returned options have no target host, the endpoint's host is used.
    /// </summary>
    public Func<SegmentEndpoint, SslClientAuthenticationOptions?>? ClientTls { get; set; }

    /// <summary>
    /// TLS for accepted connections (the worker's certificate; <see cref="SslServerAuthenticationOptions.ClientCertificateRequired"/> and a
    /// validation callback for mutual TLS), or null (the default) for plain connections.
    /// </summary>
    public SslServerAuthenticationOptions? ServerTls { get; set; }

    /// <summary>Largest frame payload accepted, in bytes; a larger length is treated as a broken stream. Default 1 GiB.</summary>
    public int MaxFrameBytes { get; set; } = 1 << 30;

    /// <summary>Socket send and receive buffer size in bytes (0 keeps the system default). Default 4 MiB.</summary>
    public int SocketBufferBytes { get; set; } = 4 << 20;
}
