using Sharplight.Runtime;

namespace Sharplight.ImageGeneration;

/// <summary>
/// Reproducible unit Gaussian noise for diffusion sampling: element i of stream s under a seed is a pure function of (seed, s, i)
/// (<see cref="CounterRandom"/>), so a seed gives the same image on every machine, provider and thread count.
/// </summary>
internal static class LatentNoise
{
    /// <summary>Elements reserved per stream (initial noise is stream 0, ancestral steps use streams 1, 2, ...).</summary>
    private const long StreamStride = 1L << 32;

    /// <summary>Draws <paramref name="count"/> values of stream <paramref name="stream"/>.</summary>
    /// <param name="seed">The image's seed.</param>
    /// <param name="stream">The stream index.</param>
    /// <param name="count">Number of values.</param>
    public static float[] Draw(long seed, long stream, int count)
    {
        ulong key = CounterRandom.Key(seed);
        var values = new float[count];
        long offset = stream * StreamStride;
        for (int i = 0; i < count; i++)
        {
            values[i] = (float)CounterRandom.Normal(key, offset + i);
        }

        return values;
    }
}
