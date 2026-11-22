using System.Buffers;
using System.Buffers.Binary;
using System.Text;

namespace Sharplight.Distributed.Protocol;

/// <summary>Builds a frame payload: little-endian integers, length-prefixed UTF-8 strings and byte blocks.</summary>
internal sealed class PayloadWriter
{
    private readonly ArrayBufferWriter<byte> _buffer;

    public PayloadWriter(int initialCapacity = 256) => _buffer = new ArrayBufferWriter<byte>(initialCapacity);

    /// <summary>The bytes written so far.</summary>
    public ReadOnlySpan<byte> WrittenSpan => _buffer.WrittenSpan;

    /// <summary>Forgets what was written, keeping the buffer.</summary>
    public void Clear() => _buffer.ResetWrittenCount();

    public PayloadWriter Byte(byte value)
    {
        _buffer.GetSpan(1)[0] = value;
        _buffer.Advance(1);
        return this;
    }

    public PayloadWriter Bool(bool value) => Byte(value ? (byte)1 : (byte)0);

    public PayloadWriter Int32(int value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(_buffer.GetSpan(4), value);
        _buffer.Advance(4);
        return this;
    }

    public PayloadWriter UInt32(uint value)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(_buffer.GetSpan(4), value);
        _buffer.Advance(4);
        return this;
    }

    public PayloadWriter Int64(long value)
    {
        BinaryPrimitives.WriteInt64LittleEndian(_buffer.GetSpan(8), value);
        _buffer.Advance(8);
        return this;
    }

    public PayloadWriter Double(double value)
    {
        BinaryPrimitives.WriteDoubleLittleEndian(_buffer.GetSpan(8), value);
        _buffer.Advance(8);
        return this;
    }

    public PayloadWriter String(string value)
    {
        int length = Encoding.UTF8.GetByteCount(value);
        Int32(length);
        Encoding.UTF8.GetBytes(value, _buffer.GetSpan(length));
        _buffer.Advance(length);
        return this;
    }

    /// <summary>A length-prefixed byte block.</summary>
    public PayloadWriter Bytes(ReadOnlySpan<byte> value)
    {
        Int32(value.Length);
        return Raw(value);
    }

    /// <summary>Bytes without a length prefix.</summary>
    public PayloadWriter Raw(ReadOnlySpan<byte> value)
    {
        value.CopyTo(_buffer.GetSpan(value.Length));
        _buffer.Advance(value.Length);
        return this;
    }

    /// <summary>A name-to-value map (symbols, scalars).</summary>
    public PayloadWriter Map(IReadOnlyDictionary<string, long> values)
    {
        Int32(values.Count);
        foreach (var (key, value) in values)
        {
            String(key);
            Int64(value);
        }

        return this;
    }

    public PayloadWriter Strings(IReadOnlyCollection<string> values)
    {
        Int32(values.Count);
        foreach (var value in values)
        {
            String(value);
        }

        return this;
    }
}
