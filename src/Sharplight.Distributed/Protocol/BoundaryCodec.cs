using System.Buffers.Binary;
using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Distributed.Protocol;

/// <summary>
/// Narrows f32 boundary values to 16 bits for the wire and widens them again (<see cref="BoundaryPrecision"/>). Which values travel narrowed
/// is decided per value by <see cref="IsNarrowed"/> on the coordinator, which tells each worker the indices of its narrowed inputs and
/// outputs with the segment, so the producer and the consumer of a value always agree, whichever way it travels.
/// </summary>
internal static class BoundaryCodec
{
    /// <summary>
    /// Whether <paramref name="value"/> travels in <paramref name="precision"/>: an f32 activation passed between segments. Graph inputs
    /// (supplied by the caller) and graph outputs (logits, hidden states the caller reads) always travel as they are.
    /// </summary>
    public static bool IsNarrowed(Value value, ComputeGraph graph, BoundaryPrecision precision)
        => precision != BoundaryPrecision.Float32 && value.Kind == ValueKind.Intermediate && value.Type == DataType.F32
            && !graph.Outputs.Values.Contains(value);

    /// <summary>Bytes of <paramref name="f32Bytes"/> bytes of f32 values on the wire.</summary>
    public static int EncodedLength(int f32Bytes, BoundaryPrecision precision)
        => precision == BoundaryPrecision.Float32 ? f32Bytes : f32Bytes / 2;

    /// <summary>Bytes of the f32 values of <paramref name="encodedBytes"/> bytes on the wire.</summary>
    public static int DecodedLength(int encodedBytes, BoundaryPrecision precision)
        => precision == BoundaryPrecision.Float32 ? encodedBytes : encodedBytes * 2;

    /// <summary>Narrows the f32 values of <paramref name="source"/> into <paramref name="destination"/> (<see cref="EncodedLength"/> bytes).</summary>
    public static void Encode(ReadOnlySpan<byte> source, Span<byte> destination, BoundaryPrecision precision)
    {
        if ((source.Length & 3) != 0 || destination.Length < EncodedLength(source.Length, precision))
        {
            throw new ArgumentException($"A boundary value of {source.Length} bytes is not a whole number of f32 values that fits {destination.Length} bytes.");
        }

        var values = MemoryMarshal.Cast<byte, float>(source);
        switch (precision)
        {
            case BoundaryPrecision.Float32:
                source.CopyTo(destination);
                break;
            case BoundaryPrecision.Float16:
            {
                var halves = MemoryMarshal.Cast<byte, Half>(destination[..(values.Length * 2)]);
                for (int i = 0; i < values.Length; i++)
                {
                    halves[i] = (Half)values[i];
                }

                if (!BitConverter.IsLittleEndian)
                {
                    SwapHalves(destination[..(values.Length * 2)]);
                }

                break;
            }

            default:
            {
                var bits = MemoryMarshal.Cast<float, uint>(values);
                for (int i = 0; i < bits.Length; i++)
                {
                    BinaryPrimitives.WriteUInt16LittleEndian(destination[(2 * i)..], ToBFloat16(bits[i]));
                }

                break;
            }
        }
    }

    /// <summary>Widens the 16-bit values of <paramref name="source"/> into f32 values in <paramref name="destination"/>.</summary>
    /// <param name="source">The values on the wire.</param>
    /// <param name="destination">Its first <see cref="DecodedLength"/> bytes (at most) are written; it may be shorter than that.</param>
    /// <param name="precision">The precision on the wire.</param>
    public static void Decode(ReadOnlySpan<byte> source, Span<byte> destination, BoundaryPrecision precision)
    {
        if (precision == BoundaryPrecision.Float32)
        {
            source[..Math.Min(source.Length, destination.Length)].CopyTo(destination);
            return;
        }

        int count = Math.Min(source.Length / 2, destination.Length / 4);
        var values = MemoryMarshal.Cast<byte, float>(destination[..(count * 4)]);
        if (precision == BoundaryPrecision.Float16)
        {
            for (int i = 0; i < count; i++)
            {
                values[i] = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(source[(2 * i)..]));
            }

            return;
        }

        var bits = MemoryMarshal.Cast<float, uint>(values);
        for (int i = 0; i < count; i++)
        {
            bits[i] = (uint)BinaryPrimitives.ReadUInt16LittleEndian(source[(2 * i)..]) << 16;
        }
    }

    /// <summary>The f32 bits rounded to bfloat16, to nearest even; NaN stays a (quiet) NaN.</summary>
    public static ushort ToBFloat16(uint bits)
    {
        if ((bits & 0x7FFF_FFFF) > 0x7F80_0000)
        {
            return (ushort)((bits >> 16) | 0x0040);
        }

        return (ushort)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
    }

    private static void SwapHalves(Span<byte> data)
    {
        var words = MemoryMarshal.Cast<byte, ushort>(data);
        BinaryPrimitives.ReverseEndianness(words, words);
    }
}
