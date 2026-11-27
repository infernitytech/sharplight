namespace Sharplight.TimeSeries;

/// <summary>
/// A covariate of a <see cref="TimeSeriesForecastRequest"/>: a related series the forecast may draw on but that is not forecast itself. Past-only covariates
/// (e.g. the sales of a related product) have only a <see cref="History"/>; known-future covariates (e.g. a promotion calendar, a weather forecast) also have
/// their values over the horizon in <see cref="Future"/>.
/// </summary>
public sealed record TimeSeriesCovariate
{
    /// <summary>The covariate's name (informational, e.g. the column header).</summary>
    public string Name { get; init; } = "";

    /// <summary>The covariate's values over the history, as long as the targets' histories and aligned with them; <see cref="float.NaN"/> marks a missing value.</summary>
    public required ReadOnlyMemory<float> History { get; init; }

    /// <summary>
    /// The covariate's known values over the horizon (one per forecast step; <see cref="float.NaN"/> marks an unknown value), or empty for a past-only
    /// covariate.
    /// </summary>
    public ReadOnlyMemory<float> Future { get; init; }
}
