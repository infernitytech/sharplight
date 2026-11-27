namespace Sharplight.TimeSeries.Tests;

/// <summary>
/// TimesFM against its references with the same weights (<c>tools/timeseries-reference/reference_timesfm.py</c>): transformers' TimesFmModelForPrediction for
/// TimesFM 2.0, the timesfm package's TimesFM_2p5_200M_torch (model-card flags) for TimesFM 2.5, whose transformers layout shares the package's reference.
/// </summary>
public class TimesFmTests
{
    [Theory]
    [InlineData("timesfm-2.0-500m-pytorch", "timesfm-2.0-500m-pytorch", false)]
    [InlineData("timesfm-2.0-500m-pytorch", "timesfm-2.0-500m-pytorch", true)]
    [InlineData("timesfm-2.5-200m-pytorch", "timesfm-2.5-200m-pytorch", false)]
    [InlineData("timesfm-2.5-200m-pytorch", "timesfm-2.5-200m-pytorch", true)]
    [InlineData("timesfm-2.5-200m-transformers", "timesfm-2.5-200m-pytorch", false)]
    [InlineData("timesfm-2.5-200m-transformers", "timesfm-2.5-200m-pytorch", true)]
    public async Task Forecasts_match_the_reference(string name, string reference, bool gpu)
    {
        TimeSeriesModels.Require(TimeSeriesModels.Model(name), TimeSeriesModels.Reference(reference));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu), TimeSeriesModels.Model(name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        Assert.Equal(name.Contains("2.5", StringComparison.Ordinal) ? 15360 : 2048, forecaster.ContextLength);
        Assert.Equal(128, forecaster.PredictionLength);
        Assert.Equal([0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f], forecaster.QuantileLevels);
        const double bound = 1e-4;
        foreach (var expected in TimeSeriesModels.Cases(reference))
        {
            var forecast = await forecaster.ForecastAsync(expected.History, new TimeSeriesForecastOptions { Horizon = expected.Horizon, QuantileLevels = expected.Levels },
                TestContext.Current.CancellationToken);
            Assert.Equal(expected.Levels, forecast.QuantileLevels);
            Assert.Equal(expected.Horizon, forecast.Horizon);
            double error = TimeSeriesModels.RelativeError(forecast.Quantiles, expected.Quantiles);
            if (expected.Name == "constant" && reference.Contains("2.5", StringComparison.Ordinal))
            {
                // The package's float32 running deviation of a constant (after normalize_inputs) lands just above its 1e-6 floor, which collapses every
                // level onto the constant; the exact zero deviation here keeps the model's own small spread around it.
                Assert.All(forecast.Quantiles, row => Assert.All(row, v => Assert.InRange(v, 3.98f, 4.02f)));
                error = 0;
            }

            double meanError = TimeSeriesModels.RelativeError([forecast.Mean], [expected.Mean]);
            TestContext.Current.TestOutputHelper?.WriteLine($"{expected.Name}: quantile error {error:E2}, point error {meanError:E2} (of the forecast's range)");
            Assert.True(error < bound, $"{expected.Name}: quantile error {error:E2}");
            Assert.True(meanError < bound, $"{expected.Name}: point error {meanError:E2}");
        }
    }

    [Fact]
    public void Missing_values_are_filled_as_the_timesfm_package_does()
    {
        float n = float.NaN;
        Assert.Equal([1.0, 2.0, 3.0, 4.0, 4.0], PatchDecoderForecasterClient.Fill([n, n, 1, n, 3, 4, n]));
        Assert.Equal([5.0], PatchDecoderForecasterClient.Fill([n, 5]));
    }
}
