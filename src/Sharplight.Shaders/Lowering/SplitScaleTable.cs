using System.Runtime.InteropServices;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// The split f16 form of f32 weight matrices for the split-f16 GEMM, computed on the host: each row is scaled by the power
/// of two that brings its largest magnitude into [2^14, 2^15), and each scaled value becomes an f16 hi part (rounded to 11
/// significant bits) and an f16 lo part (the rest), so neither part overflows or loses bits to f16 subnormals. Matches the
/// shader's <c>split_scale</c> / <c>split_h2</c> (common.slang), which split the activations the same way.
/// </summary>
internal static class SplitScaleTable
{
    /// <summary>The scale for a row whose largest magnitude is <paramref name="maxAbs"/> (NaN values are ignored by the caller).</summary>
    public static float Scale(float maxAbs)
    {
        int e = (int)((BitConverter.SingleToUInt32Bits(maxAbs) >> 23) & 0xFF);
        return BitConverter.UInt32BitsToSingle((uint)Math.Clamp(268 - e, 1, 253) << 23);
    }

    /// <summary>The scales of the <paramref name="rows"/> rows of <paramref name="k"/> f32 values in <paramref name="matrix"/>, as f32 bytes.</summary>
    /// <exception cref="ArgumentException"><paramref name="matrix"/> holds fewer than <paramref name="rows"/> x <paramref name="k"/> values.</exception>
    public static byte[] Scales(ReadOnlySpan<byte> matrix, long rows, long k)
    {
        var values = Values(matrix, rows, k);
        var scales = new float[rows];
        for (long r = 0; r < rows; r++)
        {
            scales[r] = Scale(MaxAbs(values.Slice((int)(r * k), (int)k)));
        }

        return MemoryMarshal.AsBytes(scales.AsSpan()).ToArray();
    }

    /// <summary>
    /// The split rows of <paramref name="matrix"/> (<paramref name="rows"/> x <paramref name="k"/> f32): per row the
    /// <paramref name="k"/> f16 hi parts, then (unless <paramref name="hiOnly"/>) the <paramref name="k"/> f16 lo parts of the
    /// scaled values (4 bytes per value, 2 with <paramref name="hiOnly"/>: the single-product form of
    /// <see cref="Runtime.MatrixPrecision.Fast"/>).
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="matrix"/> holds fewer than <paramref name="rows"/> x <paramref name="k"/> values.</exception>
    public static byte[] Split(ReadOnlySpan<byte> matrix, long rows, long k, bool hiOnly = false)
    {
        var values = Values(matrix, rows, k);
        long halves = hiOnly ? 1 : 2;
        var parts = new Half[rows * halves * k];
        for (long r = 0; r < rows; r++)
        {
            var row = values.Slice((int)(r * k), (int)k);
            float s = Scale(MaxAbs(row));
            var hi = parts.AsSpan((int)(r * halves * k), (int)k);
            var lo = hiOnly ? [] : parts.AsSpan((int)(r * 2 * k + k), (int)k);
            for (int i = 0; i < row.Length; i++)
            {
                float v = row[i] * s;
                float h = BitConverter.UInt32BitsToSingle((BitConverter.SingleToUInt32Bits(v) + 0x1000u) & 0xFFFFE000u);
                hi[i] = (Half)h;
                if (!hiOnly)
                {
                    lo[i] = (Half)(v - h);
                }
            }
        }

        return MemoryMarshal.AsBytes(parts.AsSpan()).ToArray();
    }

    private static ReadOnlySpan<float> Values(ReadOnlySpan<byte> matrix, long rows, long k)
    {
        var values = MemoryMarshal.Cast<byte, float>(matrix);
        if (values.Length < rows * k)
        {
            throw new ArgumentException($"Split weights need {rows} x {k} values, got {values.Length}.", nameof(matrix));
        }

        return values;
    }

    private static float MaxAbs(ReadOnlySpan<float> row)
    {
        float m = 0;
        foreach (float v in row)
        {
            float a = MathF.Abs(v);
            if (a > m)
            {
                m = a;
            }
        }

        return m;
    }
}
