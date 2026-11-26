using Sharplight.Tasks;
using Sharplight.TimeSeries;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>A forecaster that continues the last observation with a rising line, and records what it was asked.</summary>
internal sealed class FakeForecaster : ITimeSeriesForecaster
{
    public TaskClientMetadata Metadata { get; } = new(Sharplight.Models.ModelKinds.TimeSeriesForecasting, "fake-chronos", "t5");

    public int ContextLength => 512;

    public int PredictionLength => 64;

    public IReadOnlyList<float> QuantileLevels { get; } = [0.1f, 0.5f, 0.9f];

    /// <summary>When set, the forecast throws it.</summary>
    public Exception? Failure { get; set; }

    /// <summary>The histories of every request.</summary>
    public List<float[]> Histories { get; } = [];

    /// <summary>The options of every request.</summary>
    public List<TimeSeriesForecastOptions?> Requests { get; } = [];

    public Task<TimeSeriesForecast> ForecastAsync(ReadOnlyMemory<float> history, TimeSeriesForecastOptions? options = null, CancellationToken cancellationToken = default)
    {
        Histories.Add(history.ToArray());
        Requests.Add(options);
        if (Failure is { } failure)
        {
            throw failure;
        }

        int horizon = options?.Horizon ?? PredictionLength;
        var levels = options?.QuantileLevels ?? QuantileLevels;
        float last = history.Span[^1];
        float[] mean = [.. Enumerable.Range(1, horizon).Select(step => last + step)];
        var quantiles = levels.Select(level => mean.Select(value => value + ((level - 0.5f) * 10)).ToArray()).ToList();
        return Task.FromResult(new TimeSeriesForecast([.. levels], quantiles, mean));
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }
}
