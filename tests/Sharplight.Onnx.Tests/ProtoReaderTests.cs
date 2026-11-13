using Xunit;

namespace Sharplight.Onnx.Tests;

public class ProtoReaderTests
{
    private static ulong Var(byte[] b)
    {
        var r = new ProtoReader(b);
        return r.ReadVarint();
    }

    [Theory]
    [InlineData(0UL)]
    [InlineData(1UL)]
    [InlineData(127UL)]
    [InlineData(128UL)]
    [InlineData(300UL)]
    [InlineData(16384UL)]
    [InlineData(uint.MaxValue)]
    [InlineData(ulong.MaxValue)]
    public void Varint_RoundTrips(ulong v) => Assert.Equal(v, Var(new ProtoWriter().Varint(v).ToArray()));

    [Fact]
    public void NegativeInt64_UsesTenByteVarint()
    {
        var b = new ProtoWriter().Int(1, -5).ToArray();
        var r = new ProtoReader(b);
        Assert.True(r.Next(out int f, out var w));
        Assert.Equal(1, f);
        Assert.Equal(WireType.Varint, w);
        Assert.Equal(-5L, r.ReadInt64());
        Assert.False(r.Next(out _, out _));
    }

    [Fact]
    public void Fields_AllWireTypes_AndSkip()
    {
        var b = new ProtoWriter().Int(1, 150).Float(2, 1.5f).Str(3, "héllo").Bytes(4, [1, 2, 3]).Int(5, 7).ToArray();
        var r = new ProtoReader(b);
        r.Next(out _, out var w);
        Assert.Equal(150, r.ReadInt32());
        r.Next(out _, out w);
        Assert.Equal(WireType.Fixed32, w);
        Assert.Equal(1.5f, r.ReadFloat());
        r.Next(out _, out w);
        Assert.Equal("héllo", r.ReadString());
        r.Next(out int f, out w);
        Assert.Equal(4, f);
        r.Skip(w);
        r.Next(out f, out w);
        Assert.Equal(5, f);
        Assert.Equal(7, r.ReadInt32());
        Assert.True(r.AtEnd);
    }

    [Fact]
    public void ReadBytes_ReportsAbsoluteOffset()
    {
        var payload = new byte[] { 9, 8, 7 };
        var b = new ProtoWriter().Int(1, 1).Bytes(2, payload).ToArray();
        var r = new ProtoReader(b);
        r.Next(out _, out var w);
        r.ReadVarint();
        r.Next(out _, out w);
        var s = r.ReadBytes(out int off);
        Assert.Equal(payload, s.ToArray());
        Assert.Equal(payload, b.AsSpan(off, 3).ToArray());
    }

    [Fact]
    public void NestedMessage_IsBounded()
    {
        var inner = new ProtoWriter().Int(1, 42).Int(2, 43);
        var b = new ProtoWriter().Msg(1, inner).Int(2, 99).ToArray();
        var r = new ProtoReader(b);
        r.Next(out _, out _);
        var m = r.ReadMessage();
        m.Next(out _, out _);
        Assert.Equal(42, m.ReadInt32());
        m.Next(out _, out _);
        Assert.Equal(43, m.ReadInt32());
        Assert.False(m.Next(out _, out _));
        r.Next(out int f, out _);
        Assert.Equal(2, f);
        Assert.Equal(99, r.ReadInt32());
    }

    [Fact]
    public void PackedAndUnpackedRepeated()
    {
        var b = new ProtoWriter().PackedInts(1, [1, 2, -3, 1000]).Int(1, 77).PackedFloats(2, [0.5f, 2f]).Float(2, 3f).ToArray();
        var r = new ProtoReader(b);
        var ints = new List<long>();
        var floats = new List<float>();
        while (r.Next(out int f, out var w))
        {
            if (f == 1)
            {
                r.ReadRepeatedInt64(w, ints);
            }
            else
            {
                r.ReadRepeatedFloat(w, floats);
            }
        }

        Assert.Equal([1L, 2, -3, 1000, 77], ints);
        Assert.Equal([0.5f, 2f, 3f], floats);
    }

    [Fact]
    public void Truncated_Throws()
    {
        Assert.Throws<InvalidDataException>(() => { var r = new ProtoReader(new byte[] { 0x80 }); r.ReadVarint(); });
        Assert.Throws<InvalidDataException>(() => { var r = new ProtoReader(new byte[] { 0x0A, 0x05, 1, 2 }); r.Next(out _, out _); r.ReadBytes(out _); });
        Assert.Throws<InvalidDataException>(() => { var r = new ProtoReader(new byte[] { 0x0D, 1, 2 }); r.Next(out _, out _); r.ReadFixed32(); });
        Assert.Throws<InvalidDataException>(() => { var r = new ProtoReader(new byte[] { 0x00 }); r.Next(out _, out _); });   // field number 0
    }
}
