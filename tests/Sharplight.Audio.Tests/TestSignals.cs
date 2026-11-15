namespace Sharplight.Audio.Tests;

/// <summary>The analytic signal the fixtures were generated from, and comparison helpers.</summary>
internal static class TestSignals
{
    public static readonly string FixtureDirectory = Path.Combine(AppContext.BaseDirectory, "Fixtures");

    /// <summary>The fixture signal: stereo, left 0.4 sin 440 Hz + 0.2 sin 1320 Hz, right 0.4 sin 660 Hz amplitude-modulated at 3 Hz.</summary>
    public static float[] Fixture(int sampleRate, int frames)
    {
        var samples = new float[frames * 2];
        for (int i = 0; i < frames; i++)
        {
            double t = (double)i / sampleRate;
            samples[i * 2] = (float)((0.4 * Math.Sin(2 * Math.PI * 440 * t)) + (0.2 * Math.Sin(2 * Math.PI * 1320 * t)));
            samples[(i * 2) + 1] = (float)(0.4 * Math.Sin(2 * Math.PI * 660 * t) * (0.5 + (0.5 * Math.Sin(2 * Math.PI * 3 * t))));
        }

        return samples;
    }

    public static float[] Tone(int sampleRate, int channels, int frames)
    {
        var samples = new float[frames * channels];
        for (int i = 0; i < frames; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                samples[(i * channels) + c] = (float)(0.5 * Math.Sin(2 * Math.PI * (440 + (220 * c)) * i / sampleRate));
            }
        }

        return samples;
    }

    /// <summary>Signal-to-noise ratio in dB of <paramref name="actual"/> against <paramref name="reference"/> over the common length.</summary>
    public static double SnrDb(ReadOnlySpan<float> reference, ReadOnlySpan<float> actual)
    {
        int count = Math.Min(reference.Length, actual.Length);
        double signal = 0;
        double noise = 0;
        for (int i = 0; i < count; i++)
        {
            double difference = actual[i] - (double)reference[i];
            signal += reference[i] * (double)reference[i];
            noise += difference * difference;
        }

        return noise == 0 ? 200 : 10 * Math.Log10(signal / noise);
    }

    /// <summary>
    /// The best SNR over a lag of up to <paramref name="maxLagFrames"/> frames (codecs add delay): the actual signal is shifted against
    /// the reference.
    /// </summary>
    public static double BestSnrDb(float[] reference, float[] actual, int channels, int maxLagFrames, out int bestLag)
    {
        double best = double.NegativeInfinity;
        bestLag = 0;
        for (int lag = -maxLagFrames; lag <= maxLagFrames; lag++)
        {
            int shift = lag * channels;
            ReadOnlySpan<float> r = shift >= 0 ? reference.AsSpan(shift) : reference;
            ReadOnlySpan<float> a = shift >= 0 ? actual : actual.AsSpan(-shift);
            int length = Math.Min(r.Length, a.Length);
            length = Math.Min(length, reference.Length - (maxLagFrames * channels));
            double snr = SnrDb(r[..length], a[..length]);
            if (snr > best)
            {
                best = snr;
                bestLag = lag;
            }
        }

        return best;
    }
}
