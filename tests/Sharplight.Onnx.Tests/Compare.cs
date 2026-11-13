using Xunit;

namespace Sharplight.Onnx.Tests;

internal static class Compare
{
    /// <summary>|a-e| &lt;= atol + rtol*|e| with atol scaled by the output magnitude.</summary>
    public static Comparison Check(float[] actual, float[] expected, double rtol, double atolScale = 2e-4)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(1.0, expected.Select(Math.Abs).DefaultIfEmpty(0).Max());
        double atol = atolScale * scale;
        double maxAbs = 0;
        double maxRel = 0;
        bool ok = true;
        int worst = -1;
        double se = 0;
        double sd = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = Math.Abs((double)actual[i] - expected[i]);
            se += (double)expected[i] * expected[i];
            sd += d * d;
            if (double.IsNaN(actual[i]))
            {
                ok = false;
                worst = i;
                maxAbs = double.NaN;
                break;
            }
            if (d > maxAbs)
            {
                maxAbs = d;
                worst = i;
            }
            maxRel = Math.Max(maxRel, d / Math.Max(Math.Abs(expected[i]), 1e-2 * scale));
            if (d > atol + rtol * Math.Abs(expected[i]))
            {
                ok = false;
            }
        }
        return new Comparison(maxAbs, maxRel, scale, ok, worst, Math.Sqrt(sd / Math.Max(se, 1e-30)));
    }

    /// <summary>Noisy-model criterion: relative RMS error below <paramref name="rmsTol"/>.</summary>
    public static Comparison CheckRms(float[] actual, float[] expected, double rmsTol)
    {
        var c = Check(actual, expected, 1e9, 1e9);   // gathers the statistics only
        return c with { Ok = !double.IsNaN(c.MaxAbs) && c.RmsRel <= rmsTol };
    }

    public static int ArgMax(ReadOnlySpan<float> v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > v[b])
            {
                b = i;
            }
        }
        return b;
    }
}
