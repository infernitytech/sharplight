namespace Sharplight.TimeSeries;

/// <summary>
/// A multivariate forecasting task for <see cref="ITimeSeriesForecaster.ForecastAsync(TimeSeriesForecastRequest, TimeSeriesForecastOptions?, CancellationToken)"/>:
/// one or more target series that are forecast jointly, plus covariates. All series of one request form one group: models with group attention
/// (<see cref="ITimeSeriesForecaster.SupportsCovariates"/>, Chronos-2) share information across them; other models forecast every target on its own and
/// reject covariates.
/// </summary>
public sealed record TimeSeriesForecastRequest
{
    /// <summary>
    /// The target series, each oldest first at the same regular interval and all of the same length (aligned in time); <see cref="float.NaN"/> marks a missing
    /// value. Every target needs at least one observed value.
    /// </summary>
    public required IReadOnlyList<ReadOnlyMemory<float>> Targets { get; init; }

    /// <summary>The covariates (past-only or known into the future), each with a history as long as the targets'.</summary>
    public IReadOnlyList<TimeSeriesCovariate> Covariates { get; init; } = [];

    /// <summary>A request with one target series and no covariates (the univariate call).</summary>
    /// <param name="history">The target's history.</param>
    public static TimeSeriesForecastRequest Univariate(ReadOnlyMemory<float> history) => new() { Targets = [history] };
}
