namespace Sharplight.Architectures;

/// <summary>Standard normal noise (Box-Muller over a seeded <see cref="Random"/>) for the diffusion head's initial latents and the voice prompt encoder's sampling.</summary>
internal sealed class VibeVoiceNoise
{
    private readonly Random _random;

    /// <summary>Creates the generator.</summary>
    /// <param name="seed">The seed, or <see langword="null"/> for a random one.</param>
    public VibeVoiceNoise(long? seed)
    {
        _random = Tasks.Seeds.CreateRandom(seed);
    }

    /// <summary>One normal sample.</summary>
    public double Next()
    {
        double u = 1.0 - _random.NextDouble();
        double v = _random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u)) * Math.Cos(2.0 * Math.PI * v);
    }

    /// <summary><paramref name="count"/> normal samples.</summary>
    /// <param name="count">The count.</param>
    public float[] Next(int count)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (float)Next();
        }

        return values;
    }
}
