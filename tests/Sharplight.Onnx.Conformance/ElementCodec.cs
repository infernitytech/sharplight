using System.Buffers.Binary;
using Sharplight.Tensors;

namespace Sharplight.Onnx.Conformance;

/// <summary>Converts between doubles and the raw element bytes of the IR data types a compiled graph exchanges with the host.</summary>
internal static class ElementCodec
{
    public static byte[] Encode(DataType type, double[] values)
    {
        int size = type.ElementSize;
        var bytes = new byte[values.Length * size];
        for (int i = 0; i < values.Length; i++)
        {
            var s = bytes.AsSpan(i * size, size);
            double v = values[i];
            switch (type.Name)
            {
                case "f32":
                    BinaryPrimitives.WriteSingleLittleEndian(s, (float)v);
                    break;
                case "f64":
                    BinaryPrimitives.WriteDoubleLittleEndian(s, v);
                    break;
                case "f16":
                    BinaryPrimitives.WriteHalfLittleEndian(s, (Half)v);
                    break;
                case "bf16":
                    BinaryPrimitives.WriteUInt16LittleEndian(s, (ushort)(BitConverter.SingleToUInt32Bits((float)v) >> 16));
                    break;
                case "i8" or "u8" or "bool":
                    s[0] = (byte)(long)v;
                    break;
                case "i16":
                    BinaryPrimitives.WriteInt16LittleEndian(s, (short)(long)v);
                    break;
                case "i32":
                    BinaryPrimitives.WriteInt32LittleEndian(s, unchecked((int)(long)v));
                    break;
                case "u32":
                    BinaryPrimitives.WriteUInt32LittleEndian(s, unchecked((uint)(long)v));
                    break;
                case "i64":
                    BinaryPrimitives.WriteInt64LittleEndian(s, (long)v);
                    break;
                default:
                    throw new NotSupportedException($"Graph input type {type} cannot be fed by the harness");
            }
        }

        return bytes;
    }

    public static double[] Decode(DataType type, byte[] bytes)
    {
        int size = type.ElementSize;
        var values = new double[bytes.Length / size];
        for (int i = 0; i < values.Length; i++)
        {
            var s = bytes.AsSpan(i * size, size);
            values[i] = type.Name switch
            {
                "f32" => BinaryPrimitives.ReadSingleLittleEndian(s),
                "f64" => BinaryPrimitives.ReadDoubleLittleEndian(s),
                "f16" => (double)BinaryPrimitives.ReadHalfLittleEndian(s),
                "bf16" => BitConverter.UInt32BitsToSingle((uint)BinaryPrimitives.ReadUInt16LittleEndian(s) << 16),
                "i8" => (sbyte)s[0],
                "u8" or "bool" => s[0],
                "i16" => BinaryPrimitives.ReadInt16LittleEndian(s),
                "i32" => BinaryPrimitives.ReadInt32LittleEndian(s),
                "u32" => BinaryPrimitives.ReadUInt32LittleEndian(s),
                "i64" => BinaryPrimitives.ReadInt64LittleEndian(s),
                _ => throw new NotSupportedException($"Graph output type {type} cannot be read by the harness"),
            };
        }

        return values;
    }
}
