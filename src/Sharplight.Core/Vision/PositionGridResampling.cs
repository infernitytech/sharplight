namespace Sharplight.Vision;

/// <summary>
/// Resampling of learned square position-embedding grids to another grid size, as PyTorch's <c>interpolate</c> does it for DINOv2-style
/// models (<c>mode="bicubic"</c>, <c>align_corners=False</c>, no antialiasing, cubic convolution with a = -0.75).
/// </summary>
public static class PositionGridResampling
{
    private const double A = -0.75;

    /// <summary>Dense weights [<paramref name="outSize"/>, <paramref name="inSize"/>] of PyTorch's non-antialiased bicubic resize of one axis. Equal sizes give the identity.</summary>
    /// <param name="inSize">Source length.</param>
    /// <param name="outSize">Target length.</param>
    public static float[] Bicubic(int inSize, int outSize)
    {
        var m = new float[(long)outSize * inSize];
        double scale = (double)inSize / outSize;
        for (int o = 0; o < outSize; o++)
        {
            double real = scale * (o + 0.5) - 0.5;
            int i0 = (int)Math.Floor(real);
            double t = real - i0;
            Span<double> w =
            [
                Near(t + 1),
                Near(t),
                Near(1 - t),
                Far(2 - t),
            ];
            for (int k = 0; k < 4; k++)
            {
                int index = Math.Clamp(i0 - 1 + k, 0, inSize - 1);
                m[(long)o * inSize + index] += (float)w[k];
            }
        }

        return m;
    }

    /// <summary>Resamples a square grid of vectors.</summary>
    /// <param name="grid">The <c>inSize · inSize · width</c> values, row-major (row, column, channel).</param>
    /// <param name="inSize">Side of the source grid.</param>
    /// <param name="outSize">Side of the target grid.</param>
    /// <param name="width">Length of each vector.</param>
    /// <param name="antialias">True for PyTorch's <c>antialias=True</c> (the filter widens when shrinking), as DINOv2 with registers resizes its table.</param>
    /// <returns>The <c>outSize · outSize · width</c> values.</returns>
    public static float[] ResizeGrid(ReadOnlySpan<float> grid, int inSize, int outSize, int width, bool antialias = false)
    {
        if (grid.Length != (long)inSize * inSize * width)
        {
            throw new ArgumentException($"Expected {(long)inSize * inSize * width} values for a {inSize}x{inSize} grid of {width}.", nameof(grid));
        }

        var m = antialias ? ImageResampling.AntialiasedBicubic(inSize, outSize) : Bicubic(inSize, outSize);
        // Rows first: [out, in] × [in, in·width], then columns.
        var rows = new float[(long)outSize * inSize * width];
        for (int oy = 0; oy < outSize; oy++)
        {
            for (int iy = 0; iy < inSize; iy++)
            {
                float w = m[(long)oy * inSize + iy];
                if (w == 0f)
                {
                    continue;
                }

                var src = grid.Slice(iy * inSize * width, inSize * width);
                var dst = rows.AsSpan(oy * inSize * width, inSize * width);
                for (int i = 0; i < src.Length; i++)
                {
                    dst[i] += w * src[i];
                }
            }
        }

        var result = new float[(long)outSize * outSize * width];
        for (int oy = 0; oy < outSize; oy++)
        {
            for (int ox = 0; ox < outSize; ox++)
            {
                var dst = result.AsSpan((oy * outSize + ox) * width, width);
                for (int ix = 0; ix < inSize; ix++)
                {
                    float w = m[(long)ox * inSize + ix];
                    if (w == 0f)
                    {
                        continue;
                    }

                    var src = rows.AsSpan((oy * inSize + ix) * width, width);
                    for (int c = 0; c < width; c++)
                    {
                        dst[c] += w * src[c];
                    }
                }
            }
        }

        return result;
    }

    // Cubic convolution kernel for |x| in [1, 2) and [0, 1].
    private static double Far(double x) => ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;

    private static double Near(double x) => x >= 1 ? Far(x) : ((A + 2) * x - (A + 3)) * x * x + 1;
}
