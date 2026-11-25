namespace Sharplight.Audio;

/// <summary>Standard normal noise for the audio generators (Box-Muller over a <see cref="Random"/>, so a seed reproduces a run).</summary>
internal static class AudioNoise
{
    /// <summary>Fills <paramref name="values"/> with standard normal values.</summary>
    /// <param name="random">The generator.</param>
    /// <param name="values">The values to fill.</param>
    public static void Fill(Random random, Span<float> values)
    {
        for (int i = 0; i < values.Length; i += 2)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            values[i] = (float)(radius * Math.Cos(2 * Math.PI * u2));
            if (i + 1 < values.Length)
            {
                values[i + 1] = (float)(radius * Math.Sin(2 * Math.PI * u2));
            }
        }
    }

    /// <summary>A generator for a request's seed (a random one without).</summary>
    /// <param name="seed">The seed, or <see langword="null"/>.</param>
    public static Random Create(long? seed) => Tasks.Seeds.CreateRandom(seed);
}
