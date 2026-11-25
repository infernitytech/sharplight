namespace Sharplight.Architectures;

/// <summary>The VibeVoice feature extractor's loudness normalization, shared by the recognizer and the synthesizer's voice prompts.</summary>
internal static class VibeVoiceLoudness
{
    /// <summary>Scales the recording's RMS to <paramref name="targetDbfs"/>, then its peak to 1 when it exceeds it; in place.</summary>
    /// <param name="samples">The mono samples, scaled in place.</param>
    /// <param name="targetDbfs">The target RMS level in dBFS.</param>
    /// <returns><paramref name="samples"/>.</returns>
    public static float[] Normalize(float[] samples, float targetDbfs)
    {
        double energy = 0;
        foreach (float s in samples)
        {
            energy += (double)s * s;
        }

        double rms = Math.Sqrt(energy / Math.Max(1, samples.Length));
        float gain = (float)(Math.Pow(10, targetDbfs / 20) / (rms + 1e-6));
        float peak = 0;
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] *= gain;
            peak = MathF.Max(peak, MathF.Abs(samples[i]));
        }

        if (peak > 1f)
        {
            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] /= peak + 1e-6f;
            }
        }

        return samples;
    }
}
