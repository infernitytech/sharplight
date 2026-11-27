namespace Sharplight.TimeSeries.Tests;

/// <summary>One multivariate reference forecast (<c>tools/timeseries-reference/reference_chronos2.py</c>): the request, the levels and per target the quantile forecasts ([levels][horizon]) and the mean.</summary>
internal sealed record MultivariateCase(string Name, TimeSeriesForecastRequest Request, int Horizon, float[] Levels, float[][][] Quantiles, float[][] Mean)
{
    /// <inheritdoc/>
    public override string ToString() => Name;
}
