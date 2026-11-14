namespace Sharplight.Shaders.Lowering;

/// <summary>Host-side weight layout transforms, applied row by row during upload (backend-neutral).</summary>
public static class WeightRepacker
{
    /// <summary>Rows repacked together by <see cref="WeightRepack.Mxfp4Split"/>.</summary>
    public const int Mxfp4GroupRows = 16;

    /// <summary>Bytes of one repacked unit (a row, or a row group for <see cref="WeightRepack.Mxfp4Split"/>) of rows with <paramref name="k"/> logical elements.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Unknown layout.</exception>
    public static long RowBytes(WeightRepack r, long k) => r switch
    {
        WeightRepack.Q8_0Split => k / 32 * 34,
        WeightRepack.Mxfp4Split => Mxfp4GroupRows * (k / 32 * 17),
        WeightRepack.Fp8B32Split => k / 32 * 36,
        WeightRepack.Fp8B128Split => k / 128 * 132,
        _ => throw new ArgumentOutOfRangeException(nameof(r)),
    };

    /// <summary>Transforms whole units (see <see cref="RowBytes"/>) from <paramref name="src"/> into <paramref name="dst"/> (same length).</summary>
    /// <param name="r">The layout to produce.</param>
    /// <param name="k">Logical elements per row.</param>
    /// <param name="src">Source rows in the file layout.</param>
    /// <param name="dst">Receives the rows; must not overlap <paramref name="src"/>.</param>
    /// <exception cref="ArgumentException">The spans differ in length or hold partial rows.</exception>
    public static void Apply(WeightRepack r, long k, ReadOnlySpan<byte> src, Span<byte> dst)
    {
        long rowBytes = RowBytes(r, k);
        if (src.Length != dst.Length || src.Length % rowBytes != 0)
        {
            throw new ArgumentException("Repack needs whole rows.");
        }

        int blocks = (int)(k / 32);
        if (r == WeightRepack.Mxfp4Split)
        {
            for (int group = 0; group < src.Length / rowBytes; group++)
            {
                SplitMxfp4Group(src.Slice((int)(group * rowBytes), (int)rowBytes), dst.Slice((int)(group * rowBytes), (int)rowBytes), blocks);
            }

            return;
        }

        if (r is WeightRepack.Fp8B32Split or WeightRepack.Fp8B128Split)
        {
            int block = r == WeightRepack.Fp8B32Split ? 32 : 128;
            for (int row = 0; row < src.Length / rowBytes; row++)
            {
                SplitFp8Row(src.Slice((int)(row * rowBytes), (int)rowBytes), dst.Slice((int)(row * rowBytes), (int)rowBytes), (int)(k / block), block);
            }

            return;
        }

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

    // One row of { f32 d; u8 q[block] } blocks → all codes, then all scales.
    private static void SplitFp8Row(ReadOnlySpan<byte> src, Span<byte> dst, int blocks, int block)
    {
        var codes = dst[..(blocks * block)];
        var scales = dst[(blocks * block)..];
        for (int b = 0; b < blocks; b++)
        {
            var blk = src.Slice(b * (block + 4), block + 4);
            blk[..4].CopyTo(scales.Slice(b * 4, 4));
            blk[4..].CopyTo(codes.Slice(b * block, block));
        }
    }

    // One 16-row group of 17-byte blocks { u8 e; u8 qs[16] } → codes of every row (16 bytes per block), then scales.
    private static void SplitMxfp4Group(ReadOnlySpan<byte> src, Span<byte> dst, int blocks)
    {
        var codes = dst[..(Mxfp4GroupRows * blocks * 16)];
        var scales = dst[(Mxfp4GroupRows * blocks * 16)..];
        for (int row = 0; row < Mxfp4GroupRows; row++)
        {
            for (int b = 0; b < blocks; b++)
            {
                var blk = src.Slice((row * blocks + b) * 17, 17);
                scales[row * blocks + b] = blk[0];
                blk[1..].CopyTo(codes.Slice((row * blocks + b) * 16, 16));
            }
        }
    }
}
