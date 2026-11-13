using System.Buffers.Binary;
using System.Text;

namespace Sharplight.Onnx;

/// <summary>Protobuf wire types.</summary>
public enum WireType { Varint = 0, Fixed64 = 1, LengthDelimited = 2, StartGroup = 3, EndGroup = 4, Fixed32 = 5 }

/// <summary>
/// Minimal protobuf wire-format decoder over a byte span (a whole file or one embedded message).
/// Offsets reported by <see cref="ReadBytes"/> are absolute within the span, so callers can refer to
/// large payloads (tensor raw_data) without copying them.
/// </summary>
public ref struct ProtoReader
{
    private readonly ReadOnlySpan<byte> _data;
    private readonly int _end;
    private int _pos;

    public ProtoReader(ReadOnlySpan<byte> data) : this(data, 0, data.Length) { }

    public ProtoReader(ReadOnlySpan<byte> data, int start, int end)
    {
        if (start < 0 || end > data.Length || start > end)
        {
            throw new InvalidDataException("Protobuf range out of bounds.");
        }

        _data = data;
        _pos = start;
        _end = end;
    }

    public readonly int Position => _pos;
    public readonly bool AtEnd => _pos >= _end;

    /// <summary>Reads the next field key. Returns false at the end of the message.</summary>
    public bool Next(out int field, out WireType wire)
    {
        if (_pos >= _end) { field = 0; wire = 0; return false; }
        ulong key = ReadVarint();
        field = checked((int)(key >> 3));
        wire = (WireType)(key & 7);
        if (field <= 0)
        {
            throw new InvalidDataException($"Invalid protobuf field number {field} at offset {_pos}.");
        }

        return true;
    }

    public ulong ReadVarint()
    {
        ulong result = 0;
        int shift = 0;
        while (true)
        {
            if (_pos >= _end)
            {
                throw new InvalidDataException("Truncated protobuf varint.");
            }

            byte b = _data[_pos++];
            result |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0)
            {
                return result;
            }

            shift += 7;
            if (shift > 63)
            {
                throw new InvalidDataException("Protobuf varint too long.");
            }
        }
    }

    public long ReadInt64() => (long)ReadVarint();
    public int ReadInt32() => (int)(long)ReadVarint();
    public bool ReadBool() => ReadVarint() != 0;

    public uint ReadFixed32()
    {
        if (_end - _pos < 4)
        {
            throw new InvalidDataException("Truncated protobuf fixed32.");
        }

        var v = BinaryPrimitives.ReadUInt32LittleEndian(_data.Slice(_pos, 4));
        _pos += 4;
        return v;
    }

    public ulong ReadFixed64()
    {
        if (_end - _pos < 8)
        {
            throw new InvalidDataException("Truncated protobuf fixed64.");
        }

        var v = BinaryPrimitives.ReadUInt64LittleEndian(_data.Slice(_pos, 8));
        _pos += 8;
        return v;
    }

    public float ReadFloat() => BitConverter.UInt32BitsToSingle(ReadFixed32());
    public double ReadDouble() => BitConverter.UInt64BitsToDouble(ReadFixed64());

    /// <summary>Reads a length-delimited payload; <paramref name="offset"/> is its absolute start.</summary>
    public ReadOnlySpan<byte> ReadBytes(out int offset)
    {
        ulong len = ReadVarint();
        if (len > (ulong)(_end - _pos))
        {
            throw new InvalidDataException($"Protobuf length {len} exceeds the enclosing message.");
        }

        offset = _pos;
        var s = _data.Slice(_pos, (int)len);
        _pos += (int)len;
        return s;
    }

    public string ReadString() => Encoding.UTF8.GetString(ReadBytes(out _));

    /// <summary>Reader positioned over an embedded message (length-delimited field).</summary>
    public ProtoReader ReadMessage()
    {
        ulong len = ReadVarint();
        if (len > (ulong)(_end - _pos))
        {
            throw new InvalidDataException($"Protobuf length {len} exceeds the enclosing message.");
        }

        var r = new ProtoReader(_data, _pos, _pos + (int)len);
        _pos += (int)len;
        return r;
    }

    public void Skip(WireType wire)
    {
        switch (wire)
        {
            case WireType.Varint: ReadVarint(); break;
            case WireType.Fixed64: ReadFixed64(); break;
            case WireType.Fixed32: ReadFixed32(); break;
            case WireType.LengthDelimited: ReadBytes(out _); break;
            default: throw new InvalidDataException($"Unsupported protobuf wire type {wire}.");
        }
    }

    // ---- repeated scalar helpers (packed or not) ---------------------------------------------
    public void ReadRepeatedInt64(WireType wire, List<long> into)
    {
        if (wire == WireType.LengthDelimited)
        {
            var m = ReadMessage(); while (!m.AtEnd)
            {
                into.Add(m.ReadInt64());
            }
        }
        else
        {
            into.Add(ReadInt64());
        }
    }

    public void ReadRepeatedFloat(WireType wire, List<float> into)
    {
        if (wire == WireType.LengthDelimited)
        {
            var b = ReadBytes(out _);
            if (b.Length % 4 != 0)
            {
                throw new InvalidDataException("Packed float field has a length that is not a multiple of 4.");
            }

            for (int i = 0; i < b.Length; i += 4)
            {
                into.Add(BinaryPrimitives.ReadSingleLittleEndian(b.Slice(i, 4)));
            }
        }
        else
        {
            into.Add(ReadFloat());
        }
    }

    public void ReadRepeatedDouble(WireType wire, List<double> into)
    {
        if (wire == WireType.LengthDelimited)
        {
            var b = ReadBytes(out _);
            if (b.Length % 8 != 0)
            {
                throw new InvalidDataException("Packed double field has a length that is not a multiple of 8.");
            }

            for (int i = 0; i < b.Length; i += 8)
            {
                into.Add(BinaryPrimitives.ReadDoubleLittleEndian(b.Slice(i, 8)));
            }
        }
        else
        {
            into.Add(ReadDouble());
        }
    }
}
