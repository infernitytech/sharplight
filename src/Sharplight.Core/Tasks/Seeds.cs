namespace Sharplight.Tasks;

/// <summary>
/// Turns the 64-bit seeds of the public options into the 32-bit seeds of <see cref="Random"/> and the device samplers. A seed within the
/// <see cref="int"/> range maps to itself, so it reproduces the runs of the releases whose seeds were 32-bit; larger seeds fold their
/// high half into the low one.
/// </summary>
internal static class Seeds
{
    /// <summary>The 32-bit form of <paramref name="seed"/>: itself within the <see cref="int"/> range, else its halves combined.</summary>
    /// <param name="seed">The seed.</param>
    public static int ToInt32(long seed) => seed is >= int.MinValue and <= int.MaxValue ? (int)seed : unchecked((int)(seed ^ (seed >>> 32)));

    /// <summary>A generator seeded with <paramref name="seed"/> (<see cref="ToInt32"/>), or a randomly seeded one without.</summary>
    /// <param name="seed">The seed, or <see langword="null"/>.</param>
    public static Random CreateRandom(long? seed) => seed is { } s ? new Random(ToInt32(s)) : new Random();
}
