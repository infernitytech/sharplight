namespace Sharplight.Runtime.Experts;

/// <summary>
/// The decode speed to expect with routed experts read from disk (<see cref="ExpertStore.EstimateDecode"/>): bound by the reads of the selected
/// experts the cache does not hold, compute not included.
/// </summary>
/// <param name="ColdTokensPerSecond">Tokens per second when every selected expert is read (an empty cache).</param>
/// <param name="WarmTokensPerSecond">
/// Tokens per second once the cache holds its share of the experts (the cache's share of the expert bytes served without a read), or
/// <see cref="double.PositiveInfinity"/> when the cache holds all of them.
/// </param>
/// <param name="CachedShare">Share of the expert bytes the cache holds, 0 to 1.</param>
public readonly record struct ExpertDecodeEstimate(double ColdTokensPerSecond, double WarmTokensPerSecond, double CachedShare);
