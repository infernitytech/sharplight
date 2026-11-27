namespace Sharplight.TimeSeries.Tests;

/// <summary>
/// Chronos-2 against chronos-forecasting's Chronos2Pipeline with the same weights (<c>tools/timeseries-reference/reference_chronos2.py</c>): univariate
/// series, targets forecast jointly, past-only and known-future covariates, horizons that unroll.
/// </summary>
public class Chronos2Tests
{
    private const string Name = "chronos-2";

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Forecasts_match_the_reference(bool gpu)
    {
        TimeSeriesModels.Require(TimeSeriesModels.Model(Name), TimeSeriesModels.Reference(Name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu), TimeSeriesModels.Model(Name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        Assert.True(forecaster.SupportsCovariates);
        Assert.Equal(8192, forecaster.ContextLength);
        Assert.Equal(1024, forecaster.PredictionLength);
        Assert.Equal(21, forecaster.QuantileLevels.Count);
        foreach (var reference in TimeSeriesModels.MultivariateCases(Name))
        {
            var forecast = await forecaster.ForecastAsync(reference.Request, new TimeSeriesForecastOptions { Horizon = reference.Horizon, QuantileLevels = reference.Levels },
                TestContext.Current.CancellationToken);
            Assert.Equal(reference.Quantiles.Length, forecast.Targets.Count);
            for (int i = 0; i < forecast.Targets.Count; i++)
            {
                var target = forecast.Targets[i];
                Assert.Equal(reference.Levels, target.QuantileLevels);
                Assert.Equal(reference.Horizon, target.Horizon);
                double error = TimeSeriesModels.RelativeError(target.Quantiles, reference.Quantiles[i]);
                double meanError = TimeSeriesModels.RelativeError([target.Mean], [reference.Mean[i]]);
                TestContext.Current.TestOutputHelper?.WriteLine($"{reference.Name} target {i}: quantile error {error:E2}, mean error {meanError:E2} (of the forecast's range)");
                Assert.True(error < 1e-4, $"{reference.Name} target {i}: quantile error {error:E2}");
                Assert.True(meanError < 1e-4, $"{reference.Name} target {i}: mean error {meanError:E2}");
            }
        }
    }

    [Fact]
    public async Task The_univariate_call_is_a_request_with_one_target()
    {
        TimeSeriesModels.Require(TimeSeriesModels.Model(Name), TimeSeriesModels.Reference(Name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu: false), TimeSeriesModels.Model(Name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        var history = TimeSeriesModels.Cases(Name)[0].History;
        var options = new TimeSeriesForecastOptions { Horizon = 40 };
        var single = await forecaster.ForecastAsync(history, options, TestContext.Current.CancellationToken);
        var request = await forecaster.ForecastAsync(TimeSeriesForecastRequest.Univariate(history), options, TestContext.Current.CancellationToken);
        Assert.Equal(single.Quantiles, request.Targets[0].Quantiles);
        Assert.Equal(single.Mean, request.Targets[0].Mean);
    }

    [Fact]
    public async Task Requests_are_checked()
    {
        TimeSeriesModels.Require(TimeSeriesModels.Model(Name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu: false), TimeSeriesModels.Model(Name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        var token = TestContext.Current.CancellationToken;
        float[] history = [1, 2, 3, 4, 5, 6];
        await Assert.ThrowsAsync<ArgumentException>(() => forecaster.ForecastAsync(new TimeSeriesForecastRequest { Targets = [] }, cancellationToken: token));
        await Assert.ThrowsAsync<ArgumentException>(() => forecaster.ForecastAsync(new TimeSeriesForecastRequest { Targets = [history, new float[] { 1, 2 }] }, cancellationToken: token));
        await Assert.ThrowsAsync<ArgumentException>(() => forecaster.ForecastAsync(new TimeSeriesForecastRequest
        {
            Targets = [history],
            Covariates = [new TimeSeriesCovariate { Name = "short", History = new float[] { 1, 2 } }],
        }, cancellationToken: token));
        await Assert.ThrowsAsync<ArgumentException>(() => forecaster.ForecastAsync(new TimeSeriesForecastRequest
        {
            Targets = [history],
            Covariates = [new TimeSeriesCovariate { Name = "future", History = history, Future = new float[] { 1, 2, 3 } }],
        }, new TimeSeriesForecastOptions { Horizon = 4 }, token));

        // Without a horizon, the known futures set it.
        var forecast = await forecaster.ForecastAsync(new TimeSeriesForecastRequest
        {
            Targets = [history],
            Covariates = [new TimeSeriesCovariate { Name = "future", History = history, Future = new float[] { 7, 8, 9 } }],
        }, cancellationToken: token);
        Assert.Equal(3, forecast.Targets[0].Horizon);
    }

    [Fact]
    public async Task Univariate_models_forecast_targets_separately_and_reject_covariates()
    {
        string name = "chronos-bolt-tiny";
        TimeSeriesModels.Require(TimeSeriesModels.Model(name), TimeSeriesModels.Reference(name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu: false), TimeSeriesModels.Model(name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        var token = TestContext.Current.CancellationToken;
        Assert.False(forecaster.SupportsCovariates);
        var cases = TimeSeriesModels.Cases(name);
        var a = cases[0].History;
        var b = a.Select(v => (2 * v) + 1).ToArray();
        var forecast = await forecaster.ForecastAsync(new TimeSeriesForecastRequest { Targets = [a, b] }, cancellationToken: token);
        Assert.Equal((await forecaster.ForecastAsync(a, cancellationToken: token)).Quantiles, forecast.Targets[0].Quantiles);
        Assert.Equal((await forecaster.ForecastAsync(b, cancellationToken: token)).Quantiles, forecast.Targets[1].Quantiles);
        await Assert.ThrowsAsync<NotSupportedException>(() => forecaster.ForecastAsync(new TimeSeriesForecastRequest
        {
            Targets = [a],
            Covariates = [new TimeSeriesCovariate { Name = "c", History = a }],
        }, cancellationToken: token));
    }
}
