using System.Buffers.Binary;
using System.Text;

namespace Sharplight.Onnx.Tests;

/// <summary>Protobuf writer used to hand-assemble ONNX files (independent of the decoder under test).</summary>
internal sealed class ProtoWriter
{
    private readonly MemoryStream _ms = new();
    public byte[] ToArray() => _ms.ToArray();
    public int Length => (int)_ms.Length;

    public ProtoWriter Varint(ulong v)
    {
        while (v >= 0x80)
        {
            _ms.WriteByte((byte)(v | 0x80));
            v >>= 7;
        }
        _ms.WriteByte((byte)v);
        return this;
    }
    private ProtoWriter Key(int field, int wire) => Varint((ulong)((field << 3) | wire));
    public ProtoWriter Int(int field, long v) => Key(field, 0).Varint((ulong)v);
    public ProtoWriter Float(int field, float v)
    {
        Key(field, 5);
        Span<byte> b = stackalloc byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(b, v);
        _ms.Write(b);
        return this;
    }
    public ProtoWriter Bytes(int field, ReadOnlySpan<byte> b)
    {
        Key(field, 2).Varint((ulong)b.Length);
        _ms.Write(b);
        return this;
    }
    public ProtoWriter Str(int field, string s) => Bytes(field, Encoding.UTF8.GetBytes(s));
    public ProtoWriter Msg(int field, ProtoWriter m) => Bytes(field, m.ToArray());
    public ProtoWriter PackedInts(int field, IEnumerable<long> v)
    {
        var w = new ProtoWriter();
        foreach (var x in v)
        {
            w.Varint((ulong)x);
        }
        return Bytes(field, w.ToArray());
    }
    public ProtoWriter PackedFloats(int field, IEnumerable<float> v)
    {
        var l = v.ToArray();
        var b = new byte[l.Length * 4];
        for (int i = 0; i < l.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(b.AsSpan(i * 4), l[i]);
        }

        return Bytes(field, b);
    }
}
