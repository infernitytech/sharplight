namespace Sharplight.ImageGeneration;

/// <summary>Helpers of variance-preserving samples x = sqrt(ᾱ)·x₀ + sqrt(1 - ᾱ)·ε.</summary>
internal static class VariancePreserving
{
    /// <summary>Noises <paramref name="original"/> to cumulative alpha <paramref name="alpha"/>.</summary>
    public static void AddNoise(float alpha, ReadOnlySpan<float> original, ReadOnlySpan<float> noise, Span<float> sample)
    {
        float signal = MathF.Sqrt(alpha);
        float noiseLevel = MathF.Sqrt(1 - alpha);
        for (int i = 0; i < sample.Length; i++)
        {
            sample[i] = signal * original[i] + noiseLevel * noise[i];
        }
    }
}
