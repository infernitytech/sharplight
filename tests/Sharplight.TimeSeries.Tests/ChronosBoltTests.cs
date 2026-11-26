namespace Sharplight.TimeSeries.Tests;

/// <summary>Chronos-Bolt against the chronos-forecasting package's ChronosBoltPipeline with the same weights (<c>tools/timeseries-reference/reference_chronos.py</c>).</summary>
public class ChronosBoltTests
{
    [Theory]
    [InlineData("chronos-bolt-tiny", false)]
    [InlineData("chronos-bolt-tiny", true)]
    [InlineData("chronos-bolt-mini", false)]
    [InlineData("chronos-bolt-mini", true)]
    [InlineData("chronos-bolt-small", false)]
    [InlineData("chronos-bolt-small", true)]
    [InlineData("chronos-bolt-base", false)]
    [InlineData("chronos-bolt-base", true)]
    public async Task Forecasts_match_the_reference(string name, bool gpu)
    {
        TimeSeriesModels.Require(TimeSeriesModels.Model(name), TimeSeriesModels.Reference(name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu), TimeSeriesModels.Model(name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        Assert.Equal(2048, forecaster.ContextLength);
        Assert.Equal(64, forecaster.PredictionLength);
        Assert.Equal([0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f], forecaster.QuantileLevels);
        foreach (var reference in TimeSeriesModels.Cases(name))
        {
            var forecast = await forecaster.ForecastAsync(reference.History, new TimeSeriesForecastOptions { Horizon = reference.Horizon, QuantileLevels = reference.Levels },
                TestContext.Current.CancellationToken);
            Assert.Equal(reference.Levels, forecast.QuantileLevels);
            Assert.Equal(reference.Horizon, forecast.Horizon);
            double error = TimeSeriesModels.RelativeError(forecast.Quantiles, reference.Quantiles);
            double meanError = TimeSeriesModels.RelativeError([forecast.Mean], [reference.Mean]);
            TestContext.Current.TestOutputHelper?.WriteLine($"{reference.Name}: quantile error {error:E2}, mean error {meanError:E2} (of the forecast's range)");
            Assert.True(error < 1e-4, $"{reference.Name}: quantile error {error:E2}");
            Assert.True(meanError < 1e-4, $"{reference.Name}: mean error {meanError:E2}");
        }
    }

    [Fact]
    public async Task Rejects_a_history_without_observations()
    {
        string name = "chronos-bolt-tiny";
        TimeSeriesModels.Require(TimeSeriesModels.Model(name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu: false), TimeSeriesModels.Model(name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        await Assert.ThrowsAsync<ArgumentException>(() => forecaster.ForecastAsync(new[] { float.NaN, float.NaN }, cancellationToken: TestContext.Current.CancellationToken));
        await Assert.ThrowsAsync<ArgumentException>(() => forecaster.ForecastAsync(Array.Empty<float>(), cancellationToken: TestContext.Current.CancellationToken));
    }
}
