using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Text.Json;

namespace Sharplight.HuggingFace;

/// <summary>Reading helpers shared by the packed integer formats (codes in 32-bit words, scales in any float type).</summary>
internal static unsafe class HfPackedBits
{
    /// <summary>
    /// The <paramref name="bits"/>-bit code at bit <paramref name="position"/> of a little-endian stream of 32-bit words that are
    /// <paramref name="wordStride"/> words apart (codes may straddle two words, as 3-bit GPTQ codes do).
    /// </summary>
    public static int Code(byte* words, long wordStride, long position, int bits)
    {
        long word = position >> 5;
        int offset = (int)(position & 31);
        ulong value = Unsafe.ReadUnaligned<uint>(words + word * wordStride * 4) >> offset;
        if (offset + bits > 32)
        {
            value |= (ulong)Unsafe.ReadUnaligned<uint>(words + (word + 1) * wordStride * 4) << (32 - offset);
        }

        return (int)(value & ((1UL << bits) - 1));
    }

    /// <summary>Element <paramref name="index"/> of a float tensor (F16, BF16 or F32) as an exact f32.</summary>
    /// <exception cref="NotSupportedException">The tensor is not F16, BF16 or F32.</exception>
    public static float Float(SafetensorsTensor tensor, long index) => tensor.DType switch
    {
        "F16" => (float)BinaryPrimitives.ReadHalfLittleEndian(new ReadOnlySpan<byte>(tensor.Pointer + index * 2, 2)),
        "BF16" => BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(new ReadOnlySpan<byte>(tensor.Pointer + index * 2, 2)) << 16),
        "F32" => BinaryPrimitives.ReadSingleLittleEndian(new ReadOnlySpan<byte>(tensor.Pointer + index * 4, 4)),
        _ => throw new NotSupportedException($"{tensor.Name}: scales of dtype {tensor.DType} are not read."),
    };

    /// <summary>Whether every value of a float tensor is exactly an f16 value (always true for F16 tensors).</summary>
    public static bool FitsHalf(SafetensorsTensor tensor)
    {
        if (tensor.DType == "F16")
        {
            return true;
        }

        long count = tensor.Shape.Aggregate(1L, (a, d) => a * d);
        for (long i = 0; i < count; i++)
        {
            float value = Float(tensor, i);
            if ((float)(Half)value != value)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>The 32-bit integers of an I32 tensor (group indices).</summary>
    public static int[] Int32s(SafetensorsTensor tensor)
    {
        long count = tensor.Shape.Aggregate(1L, (a, d) => a * d);
        var values = new int[count];
        new ReadOnlySpan<byte>(tensor.Pointer, checked((int)(count * 4))).CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()));
        return values;
    }

    /// <summary>An integer property of a JSON object, or <paramref name="fallback"/>.</summary>
    public static int Int(JsonElement element, string name, int fallback)
        => element.ValueKind == JsonValueKind.Object && element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number
            ? value.GetInt32() : fallback;

    /// <summary>A string property of a JSON object, or null.</summary>
    public static string? String(JsonElement element, string name)
        => element.ValueKind == JsonValueKind.Object && element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() : null;
}
