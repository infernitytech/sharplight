namespace Sharplight.Architectures.SeedVr;

/// <summary>Sparse one-dimensional resampling taps: per output, the first source index, the tap count and the weights (<paramref name="Max"/> per output).</summary>
/// <param name="Start">First source index per output.</param>
/// <param name="Count">Taps per output.</param>
/// <param name="Weights">The weights, <paramref name="Max"/> per output.</param>
/// <param name="Max">Weight slots per output.</param>
internal sealed record ResampleTaps(int[] Start, int[] Count, float[] Weights, int Max);
