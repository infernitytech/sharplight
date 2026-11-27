using Sharplight.Tasks;
using Sharplight.TimeSeries;

namespace Sharplight.Serving.Tests;

/// <summary>
/// A forecaster whose forecast of every step is the series' last observed value (the mean) plus the quantile level; records the calls (batches and joint
/// requests separately).
/// </summary>
internal sealed class FakeForecaster : ITimeSeriesForecaster
{
    public TaskClientMetadata Metadata { get; } = new("time-series", "fake-bolt", "chronos_bolt");

    public int ContextLength => 512;

    public int PredictionLength => 4;

    public IReadOnlyList<float> QuantileLevels { get; } = [0.1f, 0.5f, 0.9f];

    public bool SupportsCovariates => true;

    public List<(float[][] Histories, TimeSeriesForecastOptions? Options)> Batches { get; } = [];

    public List<TimeSeriesForecastRequest> JointRequests { get; } = [];

    public Task<TimeSeriesForecast> ForecastAsync(ReadOnlyMemory<float> history, TimeSeriesForecastOptions? options = null,
        CancellationToken cancellationToken = default)
        => Task.FromResult(Forecast(history.Span, options));

    public Task<IReadOnlyList<TimeSeriesForecast>> ForecastBatchAsync(IReadOnlyList<ReadOnlyMemory<float>> histories, TimeSeriesForecastOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        TimeSeriesHistoryCheck(histories);
        Batches.Add(([.. histories.Select(h => h.ToArray())], options));
        return Task.FromResult<IReadOnlyList<TimeSeriesForecast>>([.. histories.Select(h => Forecast(h.Span, options))]);
    }

    public Task<TimeSeriesMultivariateForecast> ForecastAsync(TimeSeriesForecastRequest request, TimeSeriesForecastOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        JointRequests.Add(request);
        return Task.FromResult(new TimeSeriesMultivariateForecast([.. request.Targets.Select(t => Forecast(t.Span, options))]));
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }

    private static void TimeSeriesHistoryCheck(IReadOnlyList<ReadOnlyMemory<float>> histories)
    {
        foreach (var history in histories)
        {
            bool observed = false;
            foreach (float value in history.Span)
            {
                observed |= !float.IsNaN(value);
            }

            if (!observed)
            {
                throw new ArgumentException("The history has no observed value.", nameof(histories));
            }
        }
    }

    private TimeSeriesForecast Forecast(ReadOnlySpan<float> history, TimeSeriesForecastOptions? options)
    {
        float last = float.NaN;
        foreach (float value in history)
        {
            if (!float.IsNaN(value))
            {
                last = value;
            }
        }

        int horizon = options?.Horizon ?? PredictionLength;
        var levels = options?.QuantileLevels ?? QuantileLevels;
        var quantiles = levels.Select(level => Enumerable.Repeat(last + level, horizon).ToArray()).ToArray();
        return new TimeSeriesForecast(levels, quantiles, Enumerable.Repeat(last, horizon).ToArray());
    }
}
