using Sharplight.Distributed.Protocol;
using Sharplight.Distributed.Transport;

namespace Sharplight.Distributed.Tests;

/// <summary>The frame header, the payload encoding and the endpoint syntax.</summary>
public class FramingTests
{
    [Fact]
    public void HeaderRoundTrips()
    {
        var header = new byte[FrameHeader.Size];
        FrameHeader.Write(header, FrameKind.Execute, 0xDEADBEEF, 12345);
        var (kind, id, length) = FrameHeader.Read(header, 1 << 20);
        Assert.Equal(FrameKind.Execute, kind);
        Assert.Equal(0xDEADBEEFu, id);
        Assert.Equal(12345, length);
    }

    [Fact]
    public void MalformedHeadersAreRejected()
    {
        var header = new byte[FrameHeader.Size];
        FrameHeader.Write(header, FrameKind.Ping, 1, 100);
        Assert.Throws<InvalidDataException>(() => FrameHeader.Read(header, 99));
        header[4] = 200;
        Assert.Throws<InvalidDataException>(() => FrameHeader.Read(header, 1 << 20));
        FrameHeader.Write(header, FrameKind.Ping, 1, 1);
        header[6] = 1;
        Assert.Throws<InvalidDataException>(() => FrameHeader.Read(header, 1 << 20));
    }

    [Fact]
    public void PayloadRoundTrips()
    {
        var w = new PayloadWriter(4).Byte(7).Bool(true).Int32(-5).UInt32(9).Int64(long.MinValue).Double(1.5).String("héllo")
            .Bytes([1, 2, 3]).Map(new Dictionary<string, long> { ["T"] = 17, ["n_past"] = 3 }).Strings(["a", "bc"]);
        var r = new PayloadReader(w.WrittenSpan.ToArray());
        Assert.Equal(7, r.Byte());
        Assert.True(r.Bool());
        Assert.Equal(-5, r.Int32());
        Assert.Equal(9u, r.UInt32());
        Assert.Equal(long.MinValue, r.Int64());
        Assert.Equal(1.5, r.Double());
        Assert.Equal("héllo", r.String());
        Assert.Equal(new byte[] { 1, 2, 3 }, r.Bytes().ToArray());
        var map = new Dictionary<string, long>();
        r.Map(map);
        Assert.Equal(17, map["T"]);
        Assert.Equal(3, map["n_past"]);
        Assert.Equal(["a", "bc"], r.Strings());
        Assert.True(r.AtEnd);
    }

    [Fact]
    public void TruncatedPayloadThrows()
    {
        var bytes = new PayloadWriter().String("abcdef").WrittenSpan.ToArray();
        var r = new PayloadReader(bytes.AsMemory(0, bytes.Length - 2));
        Assert.Throws<InvalidDataException>(() => r.String());
    }

    [Theory]
    [InlineData("10.0.0.2:7070", "10.0.0.2", 7070)]
    [InlineData("node-3.cluster:1", "node-3.cluster", 1)]
    [InlineData("[::1]:7071", "::1", 7071)]
    public void EndpointsParse(string text, string host, int port)
    {
        var endpoint = SegmentEndpoint.Parse(text);
        Assert.Equal(host, endpoint.Host);
        Assert.Equal(port, endpoint.Port);
        Assert.Equal(endpoint, SegmentEndpoint.Parse(endpoint.ToString()));
    }

    [Theory]
    [InlineData("")]
    [InlineData("host")]
    [InlineData("host:")]
    [InlineData("::1:7070")]
    [InlineData("host:70000")]
    public void BadEndpointsAreRejected(string text) => Assert.False(SegmentEndpoint.TryParse(text, out _));

    [Fact]
    public void EndpointListsParse()
        => Assert.Equal([new SegmentEndpoint("a", 1), new SegmentEndpoint("b", 2)], SegmentEndpoint.ParseList(" a:1, b:2 ,"));
}
