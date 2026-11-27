namespace Sharplight.TimeSeries;

/// <summary>The result of <see cref="ITimeSeriesForecaster.ForecastAsync(TimeSeriesForecastRequest, TimeSeriesForecastOptions?, CancellationToken)"/>.</summary>
/// <param name="Targets">The forecast of every target, in the order of <see cref="TimeSeriesForecastRequest.Targets"/>.</param>
public sealed record TimeSeriesMultivariateForecast(IReadOnlyList<TimeSeriesForecast> Targets);
