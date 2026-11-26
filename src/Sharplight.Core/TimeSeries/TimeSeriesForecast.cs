namespace Sharplight.TimeSeries;

/// <summary>The result of <see cref="ITimeSeriesForecaster.ForecastAsync"/>: per quantile level, one value per forecast step.</summary>
/// <param name="QuantileLevels">The quantile levels, ascending.</param>
/// <param name="Quantiles">Per level (in the order of <paramref name="QuantileLevels"/>), the forecast of every step.</param>
/// <param name="Mean">The point forecast of every step: the mean of the samples (sampling models) or the median (quantile models).</param>
public sealed record TimeSeriesForecast(IReadOnlyList<float> QuantileLevels, IReadOnlyList<float[]> Quantiles, float[] Mean)
{
    /// <summary>The forecast steps.</summary>
    public int Horizon => Mean.Length;

    /// <summary>The forecast at one of <see cref="QuantileLevels"/>.</summary>
    /// <param name="level">The level (within 1e-6 of one of <see cref="QuantileLevels"/>).</param>
    /// <exception cref="ArgumentOutOfRangeException">The forecast has no such level.</exception>
    public float[] Quantile(float level)
    {
        for (int i = 0; i < QuantileLevels.Count; i++)
        {
            if (MathF.Abs(QuantileLevels[i] - level) < 1e-6f)
            {
                return Quantiles[i];
            }
        }

        throw new ArgumentOutOfRangeException(nameof(level), $"The forecast has no quantile level {level}.");
    }
}
