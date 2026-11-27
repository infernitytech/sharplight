namespace Sharplight.AspNetCore;

/// <summary>A name under which the container's <see cref="TimeSeries.ITimeSeriesForecaster"/> is served, resolved when the endpoint is mapped.</summary>
/// <param name="Name">The model name clients pass.</param>
internal sealed record PendingTimeSeriesForecaster(string Name);
