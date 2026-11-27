namespace Sharplight.TimeSeries;

/// <summary>Per-request settings of <see cref="ITimeSeriesForecaster"/>.</summary>
public sealed record TimeSeriesForecastOptions
{
    /// <summary>The steps to forecast, or <see langword="null"/> for the model's <see cref="ITimeSeriesForecaster.PredictionLength"/>.</summary>
    public int? Horizon { get; init; }

    /// <summary>
    /// The quantile levels of the result (each above 0 and below 1), or <see langword="null"/> for the model's
    /// <see cref="ITimeSeriesForecaster.QuantileLevels"/>.
    /// </summary>
    public IReadOnlyList<float>? QuantileLevels { get; init; }

    /// <summary>
    /// Sample paths drawn by sampling models (Chronos, Moirai 1.x); the quantiles are their empirical quantiles. <see langword="null"/>: the model's default
    /// (Chronos 20, Moirai 1.x 100).
    /// </summary>
    public int? SampleCount { get; init; }

    /// <summary>Softmax temperature of sampling models, or <see langword="null"/> for the model's default (1).</summary>
    public float? Temperature { get; init; }

    /// <summary>Top-k cutoff of sampling models (0: none), or <see langword="null"/> for the model's default (50).</summary>
    public int? TopK { get; init; }

    /// <summary>
    /// The patch size of multi-patch-size models (Moirai 1.x: 8, 16, 32, 64 or 128), or <see langword="null"/> for the model's default (32). Other models
    /// ignore it.
    /// </summary>
    public int? PatchSize { get; init; }

    /// <summary>The random seed of sampling models, or <see langword="null"/> for a random one.</summary>
    public int? Seed { get; init; }
}
