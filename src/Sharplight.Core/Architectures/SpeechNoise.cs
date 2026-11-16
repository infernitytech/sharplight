namespace Sharplight.Architectures;

/// <summary>Reproducible random inputs of the speech graphs.</summary>
internal static class SpeechNoise
{
    /// <summary>Standard-normal values (Box-Muller over a seeded generator).</summary>
    /// <param name="count">The number of values.</param>
    /// <param name="seed">The seed.</param>
    public static float[] Gaussian(int count, int seed)
    {
        var random = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i += 2)
        {
            double radius = Math.Sqrt(-2 * Math.Log(1 - random.NextDouble()));
            double angle = 2 * Math.PI * random.NextDouble();
            values[i] = (float)(radius * Math.Cos(angle));
            if (i + 1 < count)
            {
                values[i + 1] = (float)(radius * Math.Sin(angle));
            }
        }

        return values;
    }
}
