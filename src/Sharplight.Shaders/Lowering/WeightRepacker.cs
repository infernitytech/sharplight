namespace Sharplight.Shaders.Lowering;

/// <summary>Host-side weight layout transforms, applied row by row during upload (backend-neutral).</summary>
public static class WeightRepacker
{
    public static long RowBytes(WeightRepack r, long k) => r switch
    {
        WeightRepack.Q8_0Split => k / 32 * 34,
        _ => throw new ArgumentOutOfRangeException(nameof(r)),
    };

    /// <summary>Transforms whole rows from <paramref name="src"/> into <paramref name="dst"/> (same length).</summary>
    public static void Apply(WeightRepack r, long k, ReadOnlySpan<byte> src, Span<byte> dst)
    {
        long rowBytes = RowBytes(r, k);
        if (src.Length != dst.Length || src.Length % rowBytes != 0)
        {
            throw new ArgumentException("Repack needs whole rows.");
        }

        int blocks = (int)(k / 32);
        for (int row = 0; row < src.Length / rowBytes; row++)
        {
            var s = src.Slice((int)(row * rowBytes), (int)rowBytes);
            var d = dst.Slice((int)(row * rowBytes), (int)rowBytes);
            var scales = d[..(blocks * 2)];
            var quants = d[(blocks * 2)..];
            for (int b = 0; b < blocks; b++)
            {
                var blk = s.Slice(b * 34, 34);
                scales[b * 2] = blk[0];
                scales[b * 2 + 1] = blk[1];
                blk[2..].CopyTo(quants.Slice(b * 32, 32));
            }
        }
    }
}
