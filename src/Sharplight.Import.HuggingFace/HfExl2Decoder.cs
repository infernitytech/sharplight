using System.Buffers.Binary;

namespace Sharplight.Import.HuggingFace;

/// <summary>Decodes one output row (one q_weight column) of an EXL2 module to f16 (layout in <see cref="HfExl2Weights"/>).</summary>
internal sealed unsafe class HfExl2Decoder(SafetensorsTensor codes, SafetensorsTensor scales, SafetensorsTensor maxima, int[] groupBits, int[] groupRow,
    long[] groupPacked, int[] rowGroup, int[] source, long columns)
{
    /// <summary>Writes output row <paramref name="n"/> (K f16 values) into <paramref name="destination"/>.</summary>
    public void DecodeRow(long n, Span<byte> destination)
    {
        int groups = groupBits.Length;
        var scale = new Half[groups];
        for (int g = 0; g < groups; g++)
        {
            uint word = BinaryPrimitives.ReadUInt32LittleEndian(new ReadOnlySpan<byte>(scales.Pointer + (g * columns / 8 + n / 8) * 4, 4));
            int q = (int)((word >> (int)(4 * (n % 8))) & 15) + 1;
            // exllamav2 multiplies q_scale_max by 1/256 (in f16) when it loads the module.
            Half maximum = (Half)((float)BinaryPrimitives.ReadHalfLittleEndian(new ReadOnlySpan<byte>(maxima.Pointer + g * 2L, 2)) / 256f);
            scale[g] = (Half)((float)(Half)(q * q) * (float)maximum);
        }

        for (int j = 0; j < source.Length; j++)
        {
            int row = source[j];
            int g = rowGroup[row];
            int bits = groupBits[g];
            long bit = (long)(row - groupRow[g]) * bits;
            long packedRow = groupPacked[g] + bit / 32;
            int shift = (int)(bit % 32);
            ulong low = Word(packedRow, n);
            ulong value = shift + bits > 32 ? low | ((ulong)Word(packedRow + 1, n) << 32) : low;
            int code = (int)((value >> shift) & ((1UL << bits) - 1));
            Half result = (Half)((float)(code - (1 << (bits - 1))) * (float)scale[g]);
            BinaryPrimitives.WriteHalfLittleEndian(destination[(j * 2)..], result);
        }
    }

    private uint Word(long packedRow, long n)
    {
        return BinaryPrimitives.ReadUInt32LittleEndian(new ReadOnlySpan<byte>(codes.Pointer + (packedRow * columns + n) * 4, 4));
    }
}
