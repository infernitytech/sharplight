namespace OnnxBench;

/// <summary>Difference between one output and its reference: max abs / rel error, relative RMS, and the verdict.</summary>
internal sealed record OutputComparison(string Name, int Count, double MaxAbs, double MaxRel, double Rms, double Scale, bool Ok)
{
    /// <summary>Elementwise <c>|a - e| &lt;= atol * scale + rtol * |e|</c> (scale = max(1, max |e|)), or the RMS bound.</summary>
    public static OutputComparison Compare(string name, float[] actual, float[] expected, ReferenceManifest manifest)
    {
        double scale = Math.Max(1.0, expected.Select(Math.Abs).DefaultIfEmpty(0).Max());
        double atol = manifest.AbsoluteTolerance * scale;
        double maxAbs = 0, maxRel = 0, sumExpected = 0, sumDiff = 0;
        bool ok = true;
        for (int i = 0; i < expected.Length; i++)
        {
            float a = actual[i], e = expected[i];
            if (!float.IsFinite(e))
            {
                continue;
            }

            if (!float.IsFinite(a))
            {
                return new OutputComparison(name, expected.Length, double.NaN, double.NaN, double.NaN, scale, false);
            }

            double d = Math.Abs((double)a - e);
            sumExpected += (double)e * e;
            sumDiff += d * d;
            maxAbs = Math.Max(maxAbs, d);
            maxRel = Math.Max(maxRel, d / Math.Max(Math.Abs(e), 1e-2 * scale));
            if (d > atol + manifest.RelativeTolerance * Math.Abs(e))
            {
                ok = false;
            }
        }

        double rms = Math.Sqrt(sumDiff / Math.Max(sumExpected, 1e-30));
        if (manifest.RmsTolerance is { } bound)
        {
            ok = rms <= bound;
        }

        return new OutputComparison(name, expected.Length, maxAbs, maxRel, rms, scale, ok);
    }

    /// <summary>Keeps column 4 (the score) of 6-wide detection rows.</summary>
    public static float[] ScoreColumn(float[] rows) =>
        rows.Length % 6 == 0 ? Enumerable.Range(0, rows.Length / 6).Select(r => rows[r * 6 + 4]).ToArray() : rows;
}
