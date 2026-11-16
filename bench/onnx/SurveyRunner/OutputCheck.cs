/// <summary>The deviation of an output from the onnxruntime reference.</summary>
/// <param name="MaxAbs">The largest absolute difference (NaN when a non-finite value mismatches).</param>
/// <param name="MaxRel">The largest difference relative to max(|expected|, 1e-2 * scale).</param>
/// <param name="Scale">The larger of 1 and the largest reference magnitude.</param>
/// <param name="Rms">The root of sum(diff^2) / sum(expected^2).</param>
/// <param name="Ok">Whether every element is within <c>atol * scale + rtol * |expected|</c>.</param>
internal readonly record struct OutputCheck(double MaxAbs, double MaxRel, double Scale, double Rms, bool Ok)
{
    /// <summary>Compares <paramref name="actual"/> with <paramref name="expected"/>.</summary>
    public static OutputCheck Compare(float[] actual, float[] expected, double rtol, double atolScale)
    {
        double scale = Math.Max(1.0, expected.Select(Math.Abs).DefaultIfEmpty(0).Max());
        double atol = atolScale * scale;
        double maxAbs = 0;
        double maxRel = 0;
        double sumExpected = 0;
        double sumDiff = 0;
        bool ok = true;
        for (int i = 0; i < expected.Length; i++)
        {
            float a = actual[i];
            float e = expected[i];
            if (float.IsNaN(a) || float.IsInfinity(a))
            {
                if (float.IsNaN(e) || float.IsInfinity(e) && a == e)
                {
                    continue;
                }

                return new OutputCheck(double.NaN, double.NaN, scale, double.NaN, false);
            }

            if (float.IsNaN(e) || float.IsInfinity(e))
            {
                continue;
            }

            double diff = Math.Abs((double)a - e);
            sumExpected += (double)e * e;
            sumDiff += diff * diff;
            maxAbs = Math.Max(maxAbs, diff);
            maxRel = Math.Max(maxRel, diff / Math.Max(Math.Abs(e), 1e-2 * scale));
            if (diff > atol + rtol * Math.Abs(e))
            {
                ok = false;
            }
        }

        return new OutputCheck(maxAbs, maxRel, scale, Math.Sqrt(sumDiff / Math.Max(sumExpected, 1e-30)), ok);
    }
}
