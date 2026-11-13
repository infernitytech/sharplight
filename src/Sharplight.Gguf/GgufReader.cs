using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace Sharplight.Gguf;

/// <summary>
/// Bounds-checked little-endian cursor over the memory-mapped bytes of a GGUF file. Every read validates the remaining
/// length, so truncated or hostile files fail with <see cref="InvalidDataException"/> instead of reading out of bounds.
/// </summary>
internal unsafe ref struct GgufReader
{
    /// <summary>Every GGUF string starts with a 64-bit length.</summary>
    private const int MinStringBytes = 8;

    /// <summary>Every nested array starts with a 32-bit element type and a 64-bit count.</summary>
    private const int MinArrayBytes = 12;

    private readonly byte* _data;
    private readonly long _length;
    private long _position;

    /// <summary>Creates a reader over <paramref name="length"/> bytes starting at <paramref name="data"/>.</summary>
    public GgufReader(byte* data, long length)
    {
        _data = data;
        _length = length;
        _position = 0;
    }

    /// <summary>Number of bytes consumed so far.</summary>
    public readonly long Position => _position;

    private byte* Take(long count)
    {
        if (count < 0 || _position + count > _length)
        {
            throw new InvalidDataException("Unexpected end of GGUF file.");
        }

        var start = _data + _position;
        _position += count;
        return start;
    }

    /// <summary>Reads a <c>uint32</c>.</summary>
    public uint ReadU32() => Unsafe.ReadUnaligned<uint>(Take(4));

    /// <summary>Reads a <c>uint64</c>.</summary>
    public ulong ReadU64() => Unsafe.ReadUnaligned<ulong>(Take(8));

    /// <summary>Reads a length-prefixed UTF-8 string.</summary>
    public string ReadString()
    {
        long length = checked((long)ReadU64());
        if (length > int.MaxValue)
        {
            throw new InvalidDataException("String too long.");
        }

        if (length == 0)
        {
            return "";
        }

        return Encoding.UTF8.GetString(Take(length), (int)length);
    }

    /// <summary>Reads one metadata value of the given type; arrays come back as typed CLR arrays (nested arrays as <c>object[]</c>).</summary>
    public object ReadValue(GgufValueType type)
    {
        switch (type)
        {
            case GgufValueType.UInt8: return *Take(1);
            case GgufValueType.Int8: return (sbyte)*Take(1);
            case GgufValueType.UInt16: return Unsafe.ReadUnaligned<ushort>(Take(2));
            case GgufValueType.Int16: return Unsafe.ReadUnaligned<short>(Take(2));
            case GgufValueType.UInt32: return Unsafe.ReadUnaligned<uint>(Take(4));
            case GgufValueType.Int32: return Unsafe.ReadUnaligned<int>(Take(4));
            case GgufValueType.Float32: return Unsafe.ReadUnaligned<float>(Take(4));
            case GgufValueType.Bool: return *Take(1) != 0;
            case GgufValueType.String: return ReadString();
            case GgufValueType.UInt64: return Unsafe.ReadUnaligned<ulong>(Take(8));
            case GgufValueType.Int64: return Unsafe.ReadUnaligned<long>(Take(8));
            case GgufValueType.Float64: return Unsafe.ReadUnaligned<double>(Take(8));
            case GgufValueType.Array: return ReadArray();
            default: throw new InvalidDataException($"Unknown GGUF value type {(uint)type}.");
        }
    }

    private T[] ReadBlock<T>(long count) where T : unmanaged
    {
        var array = new T[count];
        if (count > 0)
        {
            var source = new ReadOnlySpan<byte>(Take(count * sizeof(T)), checked((int)(count * sizeof(T))));
            source.CopyTo(MemoryMarshal.AsBytes(array.AsSpan()));
        }

        return array;
    }

    private object ReadArray()
    {
        var elementType = (GgufValueType)ReadU32();
        long count = checked((long)ReadU64());
        if (count < 0 || count > int.MaxValue)
        {
            throw new InvalidDataException("Array too large.");
        }

        switch (elementType)
        {
            case GgufValueType.UInt8: return ReadBlock<byte>(count);
            case GgufValueType.Int8: return ReadBlock<sbyte>(count);
            case GgufValueType.UInt16: return ReadBlock<ushort>(count);
            case GgufValueType.Int16: return ReadBlock<short>(count);
            case GgufValueType.UInt32: return ReadBlock<uint>(count);
            case GgufValueType.Int32: return ReadBlock<int>(count);
            case GgufValueType.Float32: return ReadBlock<float>(count);
            case GgufValueType.UInt64: return ReadBlock<ulong>(count);
            case GgufValueType.Int64: return ReadBlock<long>(count);
            case GgufValueType.Float64: return ReadBlock<double>(count);
            case GgufValueType.Bool: return ReadBoolArray(count);
            case GgufValueType.String: return ReadStringArray(count);
            case GgufValueType.Array: return ReadNestedArrays(count);
            default: throw new InvalidDataException($"Unknown GGUF array element type {(uint)elementType}.");
        }
    }

    private bool[] ReadBoolArray(long count)
    {
        var values = new bool[count];
        var source = new ReadOnlySpan<byte>(Take(count), (int)count);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = source[i] != 0;
        }

        return values;
    }

    private string[] ReadStringArray(long count)
    {
        // Guards against absurd counts: each string needs at least its length prefix.
        if (count * MinStringBytes > _length - _position)
        {
            throw new InvalidDataException("String array larger than file.");
        }

        var values = new string[count];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = ReadString();
        }

        return values;
    }

    private object[] ReadNestedArrays(long count)
    {
        if (count * MinArrayBytes > _length - _position)
        {
            throw new InvalidDataException("Array larger than file.");
        }

        var values = new object[count];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = ReadArray();
        }

        return values;
    }
}
