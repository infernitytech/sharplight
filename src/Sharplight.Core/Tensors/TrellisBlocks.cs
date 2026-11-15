using System.Buffers.Binary;
using Sharplight.Tensors.Grids;

namespace Sharplight.Tensors;

/// <summary>
/// Decoding of the trellis-coded weight formats (<see cref="DataType.TrellisV4K8_G128"/> and its siblings), shared by the managed
/// decoders and the CPU provider.
/// </summary>
/// <remarks>
/// A block of 128 values is <c>{ f16 d; u8 qs[N · K / 8] }</c> for N steps (32 steps of 4 values or 64 steps of 2) of K bits each.
/// Step t has a 16-bit state s: for the byte-path format (V = 4, K = 8) <c>qs[(t + 31) % 32] &lt;&lt; 8 | qs[t]</c>, otherwise the 16
/// bits of the circular N·K-bit stream (bit i = <c>(qs[i / 8] &gt;&gt; (i % 8)) &amp; 1</c>) starting at bit (N - 1 - t)·K, LSB first.
/// The low 15 bits of the state select codebook points through <c>x = (s &amp; 0x7fff) · 0x9e3779b1</c> (wrapping u32): point
/// <c>x &gt;&gt; 21</c> gives values 0-1 of the step and, for V = 4, point <c>(x &gt;&gt; 10) &amp; 2047</c> values 2-3. Each value is
/// <c>d · codebook</c> in f32 (<see cref="TrellisCodebook"/>).
/// </remarks>
internal static class TrellisBlocks
{
    /// <summary>Values per block.</summary>
    public const int BlockSize = 128;

    /// <summary>The multiplier of the state hash.</summary>
    public const uint HashMultiplier = 0x9e3779b1u;

    /// <summary>The mask of the state bits the hash reads.</summary>
    public const uint StateMask = 0x7fffu;

    private static float[]? s_points;

    /// <summary>The codebook as f32 values, point i at [2i, 2i + 1].</summary>
    public static float[] Points => s_points ??= DecodeCodebook();

    /// <summary>The values per step and bits per step of a trellis format; false for every other type.</summary>
    public static bool TryGetLayout(DataType type, out int valuesPerStep, out int bitsPerStep)
    {
        (valuesPerStep, bitsPerStep) = type.Name switch
        {
            "trellis_v4k8_g128" => (4, 8),
            "trellis_v4k6_g128" => (4, 6),
            "trellis_v4k7_g128" => (4, 7),
            "trellis_v2k5_g128" => (2, 5),
            "trellis_v2k6_g128" => (2, 6),
            "trellis_v2k7_g128" => (2, 7),
            _ => (0, 0),
        };
        return valuesPerStep != 0;
    }

    /// <summary>Whether <paramref name="type"/> is a trellis format.</summary>
    public static bool Is(DataType type) => TryGetLayout(type, out _, out _);

    /// <summary>The 16-bit state of step <paramref name="t"/>.</summary>
    /// <param name="qs">The path bytes of the block (after the scale).</param>
    /// <param name="valuesPerStep">4 or 2.</param>
    /// <param name="bitsPerStep">New bits per step.</param>
    /// <param name="t">The step.</param>
    public static uint State(ReadOnlySpan<byte> qs, int valuesPerStep, int bitsPerStep, int t)
    {
        int steps = BlockSize / valuesPerStep;
        if (bitsPerStep == 8)
        {
            return (uint)qs[(t + steps - 1) % steps] << 8 | qs[t];
        }

        int bytes = steps * bitsPerStep / 8;
        int off = (steps - 1 - t) * bitsPerStep;
        int i0 = off >> 3;
        uint w = qs[i0] | (uint)qs[(i0 + 1) % bytes] << 8 | (uint)qs[(i0 + 2) % bytes] << 16;
        return (w >> (off & 7)) & 0xffffu;
    }

    /// <summary>Decodes one block to its 128 values.</summary>
    /// <param name="type">A trellis format.</param>
    /// <param name="block">The block bytes.</param>
    /// <param name="y">The 128 values to write.</param>
    public static void DecodeBlock(DataType type, ReadOnlySpan<byte> block, Span<float> y)
    {
        if (!TryGetLayout(type, out int v, out int k))
        {
            throw new ArgumentOutOfRangeException(nameof(type), type, "not a trellis format");
        }

        var points = Points;
        float d = (float)BinaryPrimitives.ReadHalfLittleEndian(block);
        var qs = block[2..];
        int steps = BlockSize / v;
        int o = 0;
        for (int t = 0; t < steps; t++)
        {
            uint x = (State(qs, v, k, t) & StateMask) * HashMultiplier;
            int p = (int)(x >> 21);
            y[o++] = d * points[2 * p];
            y[o++] = d * points[2 * p + 1];
            if (v == 4)
            {
                p = (int)((x >> 10) & 2047);
                y[o++] = d * points[2 * p];
                y[o++] = d * points[2 * p + 1];
            }
        }
    }

    private static float[] DecodeCodebook()
    {
        var bits = TrellisCodebook.Values;
        var points = new float[bits.Length];
        for (int i = 0; i < points.Length; i++)
        {
            points[i] = (float)BitConverter.UInt16BitsToHalf(bits[i]);
        }

        return points;
    }
}
