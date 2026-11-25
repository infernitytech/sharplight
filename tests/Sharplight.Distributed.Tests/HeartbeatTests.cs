using System.Diagnostics;
using Sharplight.Distributed.Protocol;
using Sharplight.Distributed.Transport;
using Sharplight.Runtime;

namespace Sharplight.Distributed.Tests;

/// <summary>A worker that completes the handshake and then goes silent (a hung process, a lost network) fails its requests at the heartbeat timeout.</summary>
public class HeartbeatTests
{
    [Fact]
    public async Task SilentWorkerFailsWithinTheHeartbeatTimeout()
    {
        var transport = new TcpSegmentTransport();
        using var listener = transport.Listen(new SegmentEndpoint("127.0.0.1", 0));
        var silent = Task.Run(async () =>
        {
            var connection = await listener.AcceptAsync(CancellationToken.None);
            connection.Receive();
            var ack = new PayloadWriter().UInt32(SegmentProtocol.Magic).Int32(SegmentProtocol.Version);
            new SegmentWorkerInfo
            {
                Name = "silent",
                InstanceId = "x",
                ModelName = "m",
                Architecture = "a",
                Version = "0",
                Devices = [new SegmentWorkerDevice("cpu", "cpu", DeviceKind.Cpu, 1L << 30, 1L << 30, false)],
                Features = new HashSet<string>(),
            }.Write(ack);
            connection.Send(FrameKind.HelloAck, 0, ack.WrittenSpan);
            return connection;
        }, TestContext.Current.CancellationToken);
        var options = new RemoteSegmentOptions
        {
            Workers = [listener.LocalEndpoint.ToString()],
            HeartbeatInterval = TimeSpan.FromMilliseconds(200),
            HeartbeatTimeout = TimeSpan.FromSeconds(1),
            ExecuteTimeout = TimeSpan.FromSeconds(60),
        };
        using var provider = new RemoteExecutionProvider(options);
        var device = Assert.Single(provider.GetDevices());
        Assert.Equal(1L << 30, device.TotalMemory);
        using var remote = (RemoteComputeDevice)provider.OpenDevice(device);
        var watch = Stopwatch.StartNew();
        var error = Assert.Throws<SegmentWorkerException>(() => remote.GetMemoryInfo());
        Assert.Contains("heartbeat", error.Message, StringComparison.OrdinalIgnoreCase);
        Assert.InRange(watch.Elapsed.TotalSeconds, 0.5, 5);
        Assert.NotNull(remote.Fault);
        (await silent).Dispose();
    }
}
