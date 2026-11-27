using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>
/// Probabilistic time-series forecasting (kind <see cref="Models.ModelKinds.TimeSeriesForecasting"/>: Chronos, Chronos-Bolt, TimesFM, Moirai): the history of one
/// univariate series in, quantile forecasts of the next steps out; <see cref="ForecastAsync(TimeSeriesForecastRequest, TimeSeriesForecastOptions?, CancellationToken)"/> forecasts
/// several related series jointly, with covariates where the model supports them (Chronos-2). Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>);
/// the call runs the compiled graphs off the caller's thread, one request at a time.
/// </summary>
public interface ITimeSeriesForecaster : ITaskClient
{
    /// <summary>The most recent observations the model looks at; older history is ignored.</summary>
    public int ContextLength { get; }

    /// <summary>
    /// The steps one model pass predicts, the default horizon. Longer horizons are predicted autoregressively (with a quality loss the model cards warn
    /// about), except by masked-encoder models (Moirai 1.x), which predict any horizon up to their limit in one pass.
    /// </summary>
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

    /// <summary>
    /// Whether the model forecasts the targets of a <see cref="TimeSeriesForecastRequest"/> jointly and uses its covariates (Chronos-2). When false,
    /// <see cref="ForecastAsync(TimeSeriesForecastRequest, TimeSeriesForecastOptions?, CancellationToken)"/> forecasts every target on its own and rejects covariates.
    /// </summary>
    public bool SupportsCovariates => false;

    /// <summary>Forecasts the steps after the histories of a multivariate request (see <see cref="TimeSeriesForecastRequest"/>).</summary>
    /// <param name="request">The targets and covariates.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults; without a horizon, the length of the covariates' known futures (when there
    /// are any) or <see cref="PredictionLength"/>.</param>
    /// <param name="cancellationToken">Stops the forecast between model passes.</param>
    /// <returns>One forecast per target.</returns>
    /// <exception cref="ArgumentException">The request has no target, a target has no observed value, or the series' lengths disagree.</exception>
    /// <exception cref="NotSupportedException">The request has covariates and the model does not support them (<see cref="SupportsCovariates"/>).</exception>
    public async Task<TimeSeriesMultivariateForecast> ForecastAsync(TimeSeriesForecastRequest request, TimeSeriesForecastOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        TimeSeriesHistory.Validate(request, TimeSeriesHistory.Horizon(request, options, PredictionLength));
        if (request.Covariates.Count > 0)
        {
            throw new NotSupportedException($"{Metadata.ModelName} does not take covariates; it forecasts univariate series only.");
        }

        var forecasts = new TimeSeriesForecast[request.Targets.Count];
        for (int i = 0; i < forecasts.Length; i++)
        {
            forecasts[i] = await ForecastAsync(request.Targets[i], options, cancellationToken).ConfigureAwait(false);
        }

        return new TimeSeriesMultivariateForecast(forecasts);
    }
}
