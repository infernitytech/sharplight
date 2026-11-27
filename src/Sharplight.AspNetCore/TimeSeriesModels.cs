using Sharplight.TimeSeries;

namespace Sharplight.AspNetCore;

/// <summary>The forecasters a host serves on <c>/v1/timeseries/forecast</c>, by the name clients pass as <c>model</c>, and the request limits.</summary>
public sealed class TimeSeriesModels
{
    /// <summary>Forecasters by name.</summary>
    public Dictionary<string, ITimeSeriesForecaster> Forecasters { get; } = new(StringComparer.Ordinal);

    /// <summary>Forecaster used when a request names none or an unknown one while only one is registered (default: the first registered).</summary>
    public string? DefaultForecaster { get; set; }

    /// <summary>Most series in one request (<c>series</c>, plus <c>covariates</c>).</summary>
    public int MaxSeries { get; set; } = 1024;

    /// <summary>Longest accepted <c>horizon</c> (long horizons run the model autoregressively, one pass per block).</summary>
    public int MaxHorizon { get; set; } = 1024;

    /// <summary>Most sample paths a request may ask sampling models for (<c>samples</c>).</summary>
    public int MaxSamples { get; set; } = 1000;

    /// <summary>Largest accepted request body.</summary>
    public long MaxRequestBodyBytes { get; set; } = 16 * 1024 * 1024;
}
