using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// <see cref="NumpyRandom"/> against numpy 2.x: <c>np.random.default_rng(seed).integers(0, high, size=n)</c>, through the 64-bit draws (the bounds of
/// Hunyuan v4 Flash's Engram multipliers: int64 max / compressed vocabulary / 2) and the buffered 32-bit ones.
/// </summary>
public sealed class NumpyRandomTests
{
    [Theory]
    [InlineData(20014L, 9007199254740991L, new[] { 7403591982942174L, 3589937225856557L, 2191576883771873L })]
    [InlineData(121318L, 38430716820228L, new[] { 4580143223222L, 36345678348469L, 31989400687791L })]
    [InlineData(5L, 1000L, new[] { 670L, 805L, 22L, 807L })]
    [InlineData(1099511627783L, 4611686018427387904L, new[] { 3826836511160502378L, 1177387229190954397L })]
    public void Integers_MatchNumpy(long seed, long high, long[] expected)
        => Assert.Equal(expected, NumpyRandom.Integers(seed, high, expected.Length));
}
