using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// The CPU LayerNorm row kernel (one-pass f64 statistics, vectorized normalize-scale-shift) against an f64 two-pass
/// reference: row lengths around the vector widths, with and without weight / bias, and rows with a large mean.
/// </summary>
public unsafe class LayerNormTests
{
    public static TheoryData<int, float> Shapes()
    {
        var d = new TheoryData<int, float>();
        foreach (int n in new[] { 1, 3, 4, 7, 8, 15, 16, 17, 31, 33, 70, 768, 1029 })
        {
            d.Add(n, 0f);
            d.Add(n, 100f);
        }
        d.Add(384, 1e4f);
        return d;
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void MatchesF64Reference(int n, float offset)
    {
        int T = 6;
        var x = Kit.Rand(T * n, 11, 1f);
        for (int i = 0; i < x.Length; i++)
        {
            x[i] += offset;
        }
        var w = Kit.Rand(n, 12, 2f);
        var bias = Kit.Rand(n, 13);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, n);
        var wv = tg.WF("w", w, n);
        var bv = tg.WF("b", bias, n);
        var eps = Attributes.From(("eps", 1e-5f));
        tg.B.Output("wb", tg.B.Emit1("LayerNorm", [xin, wv, bv], eps));
        tg.B.Output("w", tg.B.Emit1("LayerNorm", [xin, wv], eps));
        tg.B.Output("none", tg.B.Emit1("LayerNorm", [xin], eps));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(T);
        m.Run(p);
        // The inputs are f32 values with ~1e-7 relative error around the offset, so the normalized values of a row with
        // mean M and std ~0.58 carry an inherent error of ~M * 6e-8 / 0.58; the tolerance scales with that.
        double atol = 2e-5 + Math.Abs(offset) * 4e-7;
        Kit.AssertClose(Reference(x, T, n, w, bias), Kit.Out(m, "wb", p), atol, 1e-4, "w+b");
        Kit.AssertClose(Reference(x, T, n, w, null), Kit.Out(m, "w", p), atol, 1e-4, "w");
        Kit.AssertClose(Reference(x, T, n, null, null), Kit.Out(m, "none", p), atol, 1e-4, "none");
    }

    /// <summary>The statistics and the bias-only branch, called directly (graphs always pass a weight with a bias).</summary>
    [Theory]
    [InlineData(5, 0f)]
    [InlineData(70, 100f)]
    [InlineData(1029, 1e4f)]
    public void KernelsDirect(int n, float offset)
    {
        var x = Kit.Rand(n, 21, 1f);
        for (int i = 0; i < n; i++)
        {
            x[i] += offset;
        }
        // An outlier first element (the pivot of the one-pass sums) must not cost precision.
        x[0] = offset + 50f;
        var bias = Kit.Rand(n, 22);
        var y = new float[n];
        float mean;
        float variance;
        fixed (float* px = x, pb = bias, py = y)
        {
            Simd.MeanVar(px, n, out mean, out variance);
            Simd.Normalize(px, py, mean, 1f / MathF.Sqrt(variance + 1e-5f), null, pb, n);
        }
        double rm = x.Sum(v => (double)v) / n;
        double rv = x.Sum(v => ((double)v - rm) * ((double)v - rm)) / n;
        Assert.True(Math.Abs(mean - rm) <= 1e-6 * Math.Max(1, Math.Abs(rm)), $"mean {mean} vs {rm}");
        Assert.True(Math.Abs(variance - rv) <= 1e-6 * rv, $"variance {variance} vs {rv}");
        Kit.AssertClose(Reference([.. x], 1, n, null, bias), y, 2e-5 + Math.Abs(offset) * 4e-7 * 10, 1e-4, "bias only");
    }

    private static float[] Reference(float[] x, int rows, int n, float[]? w, float[]? b)
    {
        var y = new float[rows * n];
        for (int r = 0; r < rows; r++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++)
            {
                mean += x[r * n + i];
            }
            mean /= n;
            double v = 0;
            for (int i = 0; i < n; i++)
            {
                double d = x[r * n + i] - mean;
                v += d * d;
            }
            double inv = 1 / Math.Sqrt(v / n + 1e-5);
            for (int i = 0; i < n; i++)
            {
                double o = (x[r * n + i] - mean) * inv;
                if (w != null)
                {
                    o *= w[i];
                }

                if (b != null)
                {
                    o += b[i];
                }

                y[r * n + i] = (float)o;
            }
        }
        return y;
    }
}
