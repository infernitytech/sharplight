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

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task A_batch_matches_the_reference_and_the_series_one_by_one(bool gpu)
    {
        // Every reference case once at its own horizon (the long ones run the context copies batched too), all cases batched at the longest horizon.
        string name = "chronos-bolt-tiny";
        TimeSeriesModels.Require(TimeSeriesModels.Model(name), TimeSeriesModels.Reference(name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu), TimeSeriesModels.Model(name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        var cases = TimeSeriesModels.Cases(name);
        var ct = TestContext.Current.CancellationToken;
        int horizon = cases.Max(c => c.Horizon);
        var options = new TimeSeriesForecastOptions { Horizon = horizon };
        var batch = await forecaster.ForecastBatchAsync([.. cases.Select(c => (ReadOnlyMemory<float>)c.History)], options, ct);
        Assert.Equal(cases.Count, batch.Count);
        for (int i = 0; i < cases.Count; i++)
        {
            var single = await forecaster.ForecastAsync(cases[i].History, options, ct);
            double error = TimeSeriesModels.RelativeError(batch[i].Quantiles, single.Quantiles);
            Assert.True(error < 1e-5, $"{cases[i].Name}: batch vs single error {error:E2}");
            // The first block does not depend on the horizon, so shorter references compare with the forecast's first steps (native levels only).
            var reference = cases[i];
            if (reference.Levels.SequenceEqual(forecaster.QuantileLevels) && (reference.Horizon <= forecaster.PredictionLength || reference.Horizon == horizon))
            {
                var levels = reference.Levels.Select(level => batch[i].Quantile(level)[..reference.Horizon]).ToArray();
                double referenceError = TimeSeriesModels.RelativeError(levels, reference.Quantiles);
                Assert.True(referenceError < 1e-4, $"{reference.Name}: batch vs reference error {referenceError:E2}");
            }
        }
    }

    [Fact]
    public async Task A_batch_larger_than_one_pass_keeps_every_series_apart()
    {
        // 70 series (more than MaxSeriesPerPass): scaled and shifted copies of one history, so every forecast is the same affine map of the first.
        string name = "chronos-bolt-tiny";
        TimeSeriesModels.Require(TimeSeriesModels.Model(name), TimeSeriesModels.Reference(name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu: false), TimeSeriesModels.Model(name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        var history = TimeSeriesModels.Cases(name)[0].History;
        const int Count = 70;
        var histories = Enumerable.Range(0, Count).Select(i => (ReadOnlyMemory<float>)history.Select(v => (v * (1 + i)) + (10 * i)).ToArray()).ToArray();
        var forecasts = await forecaster.ForecastBatchAsync(histories, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(Count, forecasts.Count);
        for (int i = 0; i < Count; i++)
        {
            var expected = forecasts[0].Quantiles.Select(row => row.Select(v => (v * (1 + i)) + (10 * i)).ToArray()).ToArray();
            double error = TimeSeriesModels.RelativeError(forecasts[i].Quantiles, expected);
            Assert.True(error < 1e-4, $"series {i}: error {error:E2}");
        }

        await Assert.ThrowsAsync<ArgumentException>(() => forecaster.ForecastBatchAsync([history, new[] { float.NaN }], cancellationToken: TestContext.Current.CancellationToken));
        await Assert.ThrowsAsync<ArgumentException>(() => forecaster.ForecastBatchAsync([], cancellationToken: TestContext.Current.CancellationToken));
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
