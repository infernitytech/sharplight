namespace Sharplight.TimeSeries.Tests;

/// <summary>One reference forecast: the history (NaN = missing), the horizon, the quantile levels with their forecasts ([levels][horizon]) and the mean.</summary>
internal sealed record ReferenceCase(string Name, float[] History, int Horizon, float[] Levels, float[][] Quantiles, float[] Mean)
{
    /// <inheritdoc/>
    public override string ToString() => Name;
}
