namespace Sharplight.Audio;

/// <summary>
/// Band-limited sample-rate conversion by windowed-sinc interpolation (Blackman window, 16 zero crossings per side). When
/// downsampling, the kernel is widened so that it low-passes at the target Nyquist frequency, which keeps aliasing out of speech
/// front-ends (e.g. 44.1 / 48 kHz recordings into a 16 kHz recognizer).
/// </summary>
public static class AudioResampler
{
    private const int ZeroCrossings = 16;

    /// <summary>Resamples interleaved audio.</summary>
    /// <param name="samples">Interleaved input samples (whole frames of <paramref name="channels"/>).</param>
    /// <param name="channels">Channel count (positive).</param>
    /// <param name="fromRate">Input frames per second (positive).</param>
    /// <param name="toRate">Output frames per second (positive).</param>
    /// <returns>Interleaved output samples: <c>round(frames · toRate / fromRate)</c> frames.</returns>
    /// <exception cref="ArgumentException">A rate or the channel count is not positive, or the samples are not whole frames.</exception>
    public static float[] Resample(ReadOnlySpan<float> samples, int channels, int fromRate, int toRate)
    {
        if (channels <= 0 || fromRate <= 0 || toRate <= 0)
        {
            throw new ArgumentException($"Channel count ({channels}) and rates ({fromRate}, {toRate}) must be positive.");
        }

        if (samples.Length % channels != 0)
        {
            throw new ArgumentException($"{samples.Length} samples are not whole frames of {channels} channels.");
        }

        int inFrames = samples.Length / channels;
        if (fromRate == toRate)
        {
            return samples.ToArray();
        }

        int outFrames = (int)Math.Round((double)inFrames * toRate / fromRate);
        var output = new float[outFrames * channels];
        double step = (double)fromRate / toRate;
        // Normalized cutoff relative to the input Nyquist frequency: below 1 when downsampling.
        double cutoff = Math.Min(1.0, (double)toRate / fromRate);
        double halfWidth = ZeroCrossings / cutoff;
        for (int frame = 0; frame < outFrames; frame++)
        {
            double center = frame * step;
            int first = Math.Max(0, (int)Math.Ceiling(center - halfWidth));
            int last = Math.Min(inFrames - 1, (int)Math.Floor(center + halfWidth));
            for (int channel = 0; channel < channels; channel++)
            {
                double sum = 0;
                double weightSum = 0;
                for (int k = first; k <= last; k++)
                {
                    double weight = Kernel(center - k, cutoff, halfWidth);
                    sum += weight * samples[k * channels + channel];
                    weightSum += weight;
                }

                // Normalizing by the kernel sum keeps DC gain exactly 1, also at the edges where the kernel is cut off.
                output[frame * channels + channel] = weightSum == 0 ? 0f : (float)(sum / weightSum);
            }
        }

        return output;
    }

    private static double Kernel(double distance, double cutoff, double halfWidth)
    {
        double x = distance * cutoff;
        double sinc = Math.Abs(x) < 1e-9 ? 1.0 : Math.Sin(Math.PI * x) / (Math.PI * x);
        double phase = Math.PI * (distance / halfWidth + 1.0);
        double window = 0.42 - 0.5 * Math.Cos(phase) + 0.08 * Math.Cos(2 * phase);
        return sinc * window;
    }
}
