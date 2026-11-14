using System.Buffers.Binary;
using Sharplight.Tensors;

namespace Sharplight.Runtime;

/// <summary>Reads and writes the one-element control values (conditions, trip counts, iteration numbers) host segments exchange.</summary>
internal static class HostScalars
{
    /// <summary>Whether the first element is non-zero.</summary>
    public static bool IsTrue(DataType type, ReadOnlySpan<byte> bytes)
        => type == DataType.F32 ? BinaryPrimitives.ReadSingleLittleEndian(bytes) != 0 : bytes[..(int)type.ByteSize(1)].IndexOfAnyExcept((byte)0) >= 0;

    /// <summary>The first element as a count (float values are truncated, negative values clamp to 0).</summary>
    public static long ReadCount(DataType type, ReadOnlySpan<byte> bytes)
    {
        long value = type.Name switch
        {
            "f32" => (long)Math.Min(BinaryPrimitives.ReadSingleLittleEndian(bytes), long.MaxValue),
            "i32" => BinaryPrimitives.ReadInt32LittleEndian(bytes),
            "i64" => BinaryPrimitives.ReadInt64LittleEndian(bytes),
            "u32" => BinaryPrimitives.ReadUInt32LittleEndian(bytes),
            _ => throw new NotSupportedException($"A loop trip count of type {type}"),
        };
        return Math.Max(0, value);
    }

    /// <summary>Encodes a one-element value of <paramref name="type"/> (f32, i32, i64 or a byte-sized type).</summary>
    public static byte[] Encode(DataType type, long value)
    {
        var bytes = new byte[type.ByteSize(1)];
        switch (type.Name)
        {
            case "f32":
                BinaryPrimitives.WriteSingleLittleEndian(bytes, value);
                break;
            case "i32":
                BinaryPrimitives.WriteInt32LittleEndian(bytes, checked((int)value));
                break;
            case "i64":
                BinaryPrimitives.WriteInt64LittleEndian(bytes, value);
                break;
            default:
                if (bytes.Length != 1)
                {
                    throw new NotSupportedException($"A loop control value of type {type}");
                }

                bytes[0] = (byte)value;
                break;
        }

        return bytes;
    }
}
