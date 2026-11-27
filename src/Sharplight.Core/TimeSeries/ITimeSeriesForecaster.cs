using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>
/// Probabilistic time-series forecasting (kind <see cref="Models.ModelKinds.TimeSeriesForecasting"/>: Chronos, Chronos-Bolt, TimesFM): the history of one
/// univariate series in, quantile forecasts of the next steps out. Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>);
/// the call runs the compiled graphs off the caller's thread, one request at a time.
/// </summary>
public interface ITimeSeriesForecaster : ITaskClient
{
    /// <summary>The most recent observations the model looks at; older history is ignored.</summary>
    public int ContextLength { get; }

    /// <summary>The steps one model pass predicts. Longer horizons are predicted autoregressively (with a quality loss the model cards warn about).</summary>
    public int PredictionLength { get; }

    /// <summary>The quantile levels the model predicts natively (other levels are interpolated between them).</summary>
    public IReadOnlyList<float> QuantileLevels { get; }

    /// <summary>Forecasts the steps after <paramref name="history"/>.</summary>
    /// <param name="history">The observations, oldest first, at a regular interval; <see cref="float.NaN"/> marks a missing value.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops the forecast between model passes.</param>
    /// <exception cref="ArgumentException">The history is empty or has no observed value.</exception>
    public Task<TimeSeriesForecast> ForecastAsync(ReadOnlyMemory<float> history, TimeSeriesForecastOptions? options = null,
        CancellationToken cancellationToken = default);
}
