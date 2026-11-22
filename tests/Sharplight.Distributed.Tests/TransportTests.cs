using System.Net.Security;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using Sharplight.Distributed.Transport;

namespace Sharplight.Distributed.Tests;

/// <summary>TCP frames on loopback: both directions, large payloads, refused connections and (mutual) TLS.</summary>
public class TransportTests
{
    private static readonly SegmentEndpoint Loopback = new("127.0.0.1", 0);

    [Fact]
    public async Task FramesTravelBothWays()
    {
        var transport = new TcpSegmentTransport();
        using var listener = transport.Listen(Loopback);
        var accept = listener.AcceptAsync(TestContext.Current.CancellationToken);
        using var client = await transport.ConnectAsync(listener.LocalEndpoint, TestContext.Current.CancellationToken);
        using var server = await accept;
        var large = new byte[8 << 20];
        Random.Shared.NextBytes(large);
        client.Send(FrameKind.Execute, 42, large);
        client.Send(FrameKind.Ping, 0, []);
        var first = server.Receive();
        Assert.Equal(FrameKind.Execute, first.Kind);
        Assert.Equal(42u, first.RequestId);
        Assert.True(first.Payload.Span.SequenceEqual(large));
        Assert.Equal(FrameKind.Ping, server.Receive().Kind);
        server.Send(FrameKind.Pong, 7, [1, 2, 3]);
        var reply = client.Receive();
        Assert.Equal(FrameKind.Pong, reply.Kind);
        Assert.Equal(new byte[] { 1, 2, 3 }, reply.Payload.ToArray());
    }

    [Fact]
    public async Task ClosedConnectionFailsTheReceiver()
    {
        var transport = new TcpSegmentTransport();
        using var listener = transport.Listen(Loopback);
        var accept = listener.AcceptAsync(TestContext.Current.CancellationToken);
        var client = await transport.ConnectAsync(listener.LocalEndpoint, TestContext.Current.CancellationToken);
        using var server = await accept;
        client.Dispose();
        Assert.Throws<IOException>(() => server.Receive());
    }

    [Fact]
    public async Task OversizedFramesBreakTheStream()
    {
        var transport = new TcpSegmentTransport(new TcpSegmentTransportOptions { MaxFrameBytes = 1024 });
        using var listener = transport.Listen(Loopback);
        var accept = listener.AcceptAsync(TestContext.Current.CancellationToken);
        using var client = await new TcpSegmentTransport().ConnectAsync(listener.LocalEndpoint, TestContext.Current.CancellationToken);
        using var server = await accept;
        client.Send(FrameKind.Execute, 1, new byte[4096]);
        Assert.Throws<IOException>(() => server.Receive());
    }

    [Fact]
    public async Task UnreachableWorkerFailsQuickly()
    {
        var transport = new TcpSegmentTransport(new TcpSegmentTransportOptions { ConnectTimeout = TimeSpan.FromSeconds(2) });
        int port;
        using (var probe = transport.Listen(Loopback))
        {
            port = probe.LocalEndpoint.Port;
        }

        var started = DateTime.UtcNow;
        await Assert.ThrowsAsync<IOException>(() => transport.ConnectAsync(new SegmentEndpoint("127.0.0.1", port), TestContext.Current.CancellationToken));
        Assert.True(DateTime.UtcNow - started < TimeSpan.FromSeconds(5));
    }

    [Fact]
    public async Task MutualTlsCarriesFrames()
    {
        using var serverCertificate = SelfSigned("CN=worker");
        using var clientCertificate = SelfSigned("CN=coordinator");
        string? seenClient = null;
        var server = new TcpSegmentTransport(new TcpSegmentTransportOptions
        {
            ServerTls = new SslServerAuthenticationOptions
            {
                ServerCertificate = serverCertificate,
                ClientCertificateRequired = true,
                RemoteCertificateValidationCallback = (_, certificate, _, _) =>
                {
                    seenClient = certificate?.Subject;
                    return certificate?.Subject == "CN=coordinator";
                },
            },
        });
        var client = new TcpSegmentTransport(new TcpSegmentTransportOptions
        {
            ClientTls = _ => new SslClientAuthenticationOptions
            {
                TargetHost = "worker",
                ClientCertificates = [clientCertificate],
                RemoteCertificateValidationCallback = (_, certificate, _, _) => certificate?.Subject == "CN=worker",
            },
        });
        using var listener = server.Listen(Loopback);
        var accept = listener.AcceptAsync(TestContext.Current.CancellationToken);
        using var connection = await client.ConnectAsync(listener.LocalEndpoint, TestContext.Current.CancellationToken);
        using var accepted = await accept;
        connection.Send(FrameKind.Hello, 3, [9, 8, 7]);
        var frame = accepted.Receive();
        Assert.Equal(FrameKind.Hello, frame.Kind);
        Assert.Equal(new byte[] { 9, 8, 7 }, frame.Payload.ToArray());
        Assert.Equal("CN=coordinator", seenClient);
    }

    [Fact]
    public async Task TlsWorkerRefusesAPlainClient()
    {
        using var serverCertificate = SelfSigned("CN=worker");
        var server = new TcpSegmentTransport(new TcpSegmentTransportOptions
        {
            ConnectTimeout = TimeSpan.FromSeconds(3),
            ServerTls = new SslServerAuthenticationOptions { ServerCertificate = serverCertificate, ClientCertificateRequired = true },
        });
        using var listener = server.Listen(Loopback);
        var accept = listener.AcceptAsync(TestContext.Current.CancellationToken);
        using var plain = await new TcpSegmentTransport().ConnectAsync(listener.LocalEndpoint, TestContext.Current.CancellationToken);
        plain.Send(FrameKind.Hello, 1, [1]);
        await Assert.ThrowsAnyAsync<Exception>(() => accept);
    }

    private static X509Certificate2 SelfSigned(string subject)
    {
        using var key = ECDsa.Create(ECCurve.NamedCurves.nistP256);
        var request = new CertificateRequest(subject, key, HashAlgorithmName.SHA256);
        using var certificate = request.CreateSelfSigned(DateTimeOffset.UtcNow.AddMinutes(-5), DateTimeOffset.UtcNow.AddHours(1));
        return X509CertificateLoader.LoadPkcs12(certificate.Export(X509ContentType.Pfx), null);
    }
}
