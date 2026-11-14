namespace Sharplight.Runtime;

/// <summary>
/// The deterministic random generator of the IR random ops (<c>Random</c>, <c>Multinomial</c>). It is counter based: the value of
/// element i depends only on the op's seed and i, so every provider, thread split and run produces the same tensor for the same seed.
/// </summary>
/// <remarks>
/// <para>Bits of draw k: <c>Mix(Mix(seed) + (k + 1) · 0x9E3779B97F4A7C15)</c>, where Mix is the SplitMix64 finalizer.</para>
/// <para>Uniform [0, 1): the upper 24 bits of draw i, times 2^-24 (exactly representable in f32).</para>
/// <para>Normal: Box-Muller on draws 2i and 2i + 1 (53-bit uniforms u1 ∈ (0, 1], u2 ∈ [0, 1)): sqrt(-2 ln u1) · cos(2π u2).</para>
/// </remarks>
internal static class CounterRandom
{
    private const ulong Golden = 0x9E3779B97F4A7C15UL;

    /// <summary>Key of a seed (scrambled once so that neighbouring seeds give unrelated streams).</summary>
    public static ulong Key(long seed) => Mix(unchecked((ulong)seed));

    /// <summary>64 random bits of draw <paramref name="index"/>.</summary>
    public static ulong Bits(ulong key, long index) => Mix(unchecked(key + ((ulong)index + 1) * Golden));

    /// <summary>Uniform float in [0, 1) for element <paramref name="index"/>.</summary>
    public static float Uniform(ulong key, long index) => (Bits(key, index) >> 40) * (1.0f / (1 << 24));

    /// <summary>Standard normal sample for element <paramref name="index"/>.</summary>
    public static double Normal(ulong key, long index)
    {
        double u1 = ((Bits(key, 2 * index) >> 11) + 1) * (1.0 / (1UL << 53));
        double u2 = (Bits(key, (2 * index) + 1) >> 11) * (1.0 / (1UL << 53));
        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    private static ulong Mix(ulong z)
    {
        unchecked
        {
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }
}
