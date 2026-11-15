namespace Sharplight.Vision;

/// <summary>
/// One-dimensional resampling weights with PyTorch's <c>interpolate</c> semantics (align_corners off), used to resize images and learned
/// position grids or tables. A 2-D resize is separable: rows, then columns.
/// </summary>
public static class ImageResampling
{
    /// <summary>
    /// Dense weights [<paramref name="outSize"/>, <paramref name="inSize"/>] of an antialiased bicubic resize (PIL, and PyTorch's
    /// <c>mode="bicubic", antialias=True</c>; Keys kernel a = −0.5, widened by the scale when shrinking). Equal sizes give the identity.
    /// </summary>
    /// <param name="inSize">Source length.</param>
    /// <param name="outSize">Target length.</param>
    public static float[] AntialiasedBicubic(int inSize, int outSize)
    {
        var (start, count, w, max) = Coefficients(inSize, outSize);
        var m = new float[(long)outSize * inSize];
        for (int o = 0; o < outSize; o++)
        {
            for (int i = 0; i < count[o]; i++)
            {
                m[(long)o * inSize + start[o] + i] = w[o * max + i];
            }
        }

        return m;
    }

    /// <summary>
    /// Dense weights [<paramref name="outSize"/>, <paramref name="inSize"/>] of PyTorch's <c>mode="linear"</c> resize (two taps, source
    /// coordinate (o + 0.5)·in/out − 0.5 clamped at 0, no antialiasing). Equal sizes give the identity.
    /// </summary>
    /// <param name="inSize">Source length.</param>
    /// <param name="outSize">Target length.</param>
    public static float[] Linear(int inSize, int outSize)
    {
        var m = new float[(long)outSize * inSize];
        double scale = (double)inSize / outSize;
        for (int o = 0; o < outSize; o++)
        {
            double src = Math.Max((o + 0.5) * scale - 0.5, 0);
            int i0 = Math.Min((int)Math.Floor(src), inSize - 1);
            int i1 = Math.Min(i0 + 1, inSize - 1);
            double l1 = src - i0;
            m[(long)o * inSize + i0] += (float)(1 - l1);
            m[(long)o * inSize + i1] += (float)l1;
        }

        return m;
    }

    private static double Cubic(double x)
    {
        const double a = -0.5;
        x = Math.Abs(x);
        if (x < 1)
        {
            return ((a + 2) * x - (a + 3)) * x * x + 1;
        }

        if (x < 2)
        {
            return (((x - 5) * x + 8) * x - 4) * a;
        }

        return 0;
    }

    /// <summary>Sparse antialiased bicubic (or bilinear) taps: per output, the first source index, the tap count and the weights (Max per output).</summary>
    internal static (int[] Start, int[] Count, float[] Weights, int Max) Coefficients(int inSize, int outSize, ImageResampleFilter filter = ImageResampleFilter.Bicubic)
    {
        double scale = (double)inSize / outSize;
        double filterScale = Math.Max(scale, 1.0);
        bool bilinear = filter == ImageResampleFilter.Bilinear;
        double support = (bilinear ? 1.0 : 2.0) * filterScale;
        int max = (int)Math.Ceiling(support) * 2 + 1;
        var start = new int[outSize];
        var count = new int[outSize];
        var w = new float[outSize * max];
        for (int x = 0; x < outSize; x++)
        {
            double center = (x + 0.5) * scale;
            int xmin = Math.Max((int)(center - support + 0.5), 0);
            int xmax = Math.Min((int)(center + support + 0.5), inSize);
            double sum = 0;
            var tmp = new double[xmax - xmin];
            for (int i = 0; i < tmp.Length; i++)
            {
                tmp[i] = bilinear ? Math.Max(0, 1 - Math.Abs((xmin + i - center + 0.5) / filterScale)) : Cubic((xmin + i - center + 0.5) / filterScale);
                sum += tmp[i];
            }
            start[x] = xmin;
            count[x] = tmp.Length;
            for (int i = 0; i < tmp.Length; i++)
            {
                w[x * max + i] = (float)(tmp[i] / sum);
            }
        }
        return (start, count, w, max);
    }
}
