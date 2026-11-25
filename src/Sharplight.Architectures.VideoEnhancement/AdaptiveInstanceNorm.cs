namespace Sharplight.Architectures.VideoEnhancement;

/// <summary>
/// Adaptive instance normalization of planar clips (the <c>adain_color_fix</c> of SeedVR's colour_fix.py): every frame and channel of the
/// content gets the mean and standard deviation of the same frame and channel of the reference (unbiased variance plus 1e-5).
/// </summary>
internal static class AdaptiveInstanceNorm
{
    private const double Eps = 1e-5;

    /// <summary>Normalizes <paramref name="content"/> in place.</summary>
    /// <param name="content">Frames [F, 3, plane].</param>
    /// <param name="reference">Frames of the same layout.</param>
    /// <param name="frames">Frames F.</param>
    /// <param name="plane">Pixels per channel.</param>
    public static void Apply(float[] content, float[] reference, int frames, int plane)
    {
        Parallel.For(0, frames * 3, index =>
        {
            var target = content.AsSpan(index * plane, plane);
            var (contentMean, contentStd) = Statistics(target);
            var (referenceMean, referenceStd) = Statistics(reference.AsSpan(index * plane, plane));
            double gain = referenceStd / contentStd;
            for (int i = 0; i < target.Length; i++)
            {
                target[i] = (float)((target[i] - contentMean) * gain + referenceMean);
            }
        });
    }

    private static (double Mean, double Std) Statistics(ReadOnlySpan<float> values)
    {
        double sum = 0;
        foreach (float v in values)
        {
            sum += v;
        }

        double mean = sum / values.Length;
        double squares = 0;
        foreach (float v in values)
        {
            squares += (v - mean) * (v - mean);
        }

        return (mean, Math.Sqrt(squares / Math.Max(1, values.Length - 1) + Eps));
    }
}
