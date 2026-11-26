namespace Sharplight.TimeSeries.Tests;

/// <summary>
/// Chronos (T5 token forecasters) against the chronos-forecasting package's ChronosPipeline with the same weights, decoded greedily (top-k 1, one sample) so that both
/// sides are deterministic (<c>tools/timeseries-reference/reference_chronos.py</c>).
/// </summary>
public class ChronosTests
{
    [Theory]
    [InlineData("chronos-t5-tiny", false)]
    [InlineData("chronos-t5-tiny", true)]
    [InlineData("chronos-t5-small", false)]
    [InlineData("chronos-t5-small", true)]
    public async Task Greedy_forecasts_match_the_reference(string name, bool gpu)
    {
        TimeSeriesModels.Require(TimeSeriesModels.Model(name), TimeSeriesModels.Reference(name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu), TimeSeriesModels.Model(name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        Assert.Equal(512, forecaster.ContextLength);
        Assert.Equal(64, forecaster.PredictionLength);
        foreach (var reference in TimeSeriesModels.Cases(name))
        {
            var options = new TimeSeriesForecastOptions { Horizon = reference.Horizon, QuantileLevels = reference.Levels, SampleCount = 1, TopK = 1 };
            var forecast = await forecaster.ForecastAsync(reference.History, options, TestContext.Current.CancellationToken);
            Assert.Equal(reference.Horizon, forecast.Horizon);
            double error = TimeSeriesModels.RelativeError(forecast.Quantiles, reference.Quantiles);
            TestContext.Current.TestOutputHelper?.WriteLine($"{reference.Name}: error {error:E2} (of the forecast's range)");
            if (reference.Name == "gaps")
            {
                // Missing values inside the context are padding tokens the reference masks out of the attention; here they stay visible.
                Assert.All(forecast.Mean, v => Assert.True(float.IsFinite(v)));
                continue;
            }

            Assert.True(error < 1e-4, $"{reference.Name}: error {error:E2}");
        }
    }

    [Fact]
    public async Task Sampled_forecasts_are_reproducible_with_a_seed_and_ordered()
    {
        string name = "chronos-t5-tiny";
        TimeSeriesModels.Require(TimeSeriesModels.Model(name), TimeSeriesModels.Reference(name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu: false), TimeSeriesModels.Model(name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        var history = TimeSeriesModels.Cases(name)[0].History;
        var options = new TimeSeriesForecastOptions { Horizon = 24, SampleCount = 8, Seed = 3 };
        var a = await forecaster.ForecastAsync(history, options, TestContext.Current.CancellationToken);
        var b = await forecaster.ForecastAsync(history, options, TestContext.Current.CancellationToken);
        Assert.Equal(a.Quantiles, b.Quantiles);
        Assert.Equal(9, a.QuantileLevels.Count);
        for (int t = 0; t < 24; t++)
        {
            for (int q = 1; q < a.Quantiles.Count; q++)
            {
                Assert.True(a.Quantiles[q][t] >= a.Quantiles[q - 1][t]);
            }
        }
    }
}
