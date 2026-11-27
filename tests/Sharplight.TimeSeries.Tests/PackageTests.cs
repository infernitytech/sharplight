using Microsoft.Extensions.DependencyInjection;
using Sharplight.Models;
using Sharplight.Package;

namespace Sharplight.TimeSeries.Tests;

/// <summary>The forecasters in <c>.slm</c> packages: converted once, loaded again, the same forecasts as the checkpoint.</summary>
public class PackageTests
{
    [Theory]
    [InlineData("chronos-bolt-tiny")]
    [InlineData("chronos-t5-tiny")]
    [InlineData("timesfm-2.5-200m-pytorch")]
    [InlineData("moirai-2.0-R-small")]
    [InlineData("moirai-1.1-R-small")]
    [InlineData("chronos-2")]
    public async Task Package_forecasts_like_the_checkpoint(string name)
    {
        TimeSeriesModels.Require(TimeSeriesModels.Model(name), TimeSeriesModels.Reference(name));
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "timeseries-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"{name}-{Guid.NewGuid():N}.slm");
        try
        {
            TimeSeriesModels.Services.GetRequiredService<ModelConverter>().Convert(TimeSeriesModels.Model(name), path);
            Assert.Equal(ModelKinds.TimeSeriesForecasting, ModelPackageReader.Open(path).Definition.Description!.Kind);
            var history = TimeSeriesModels.Cases(name)[0].History;
            var options = new TimeSeriesForecastOptions { Horizon = 80, SampleCount = 4, Seed = 11 };
            using var original = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu: false), TimeSeriesModels.Model(name));
            using var packaged = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu: false), path);
            using var a = TimeSeriesModels.Client<ITimeSeriesForecaster>(original);
            using var b = TimeSeriesModels.Client<ITimeSeriesForecaster>(packaged);
            Assert.Equal(a.ContextLength, b.ContextLength);
            Assert.Equal(a.QuantileLevels, b.QuantileLevels);
            var expected = await a.ForecastAsync(history, options, TestContext.Current.CancellationToken);
            var actual = await b.ForecastAsync(history, options, TestContext.Current.CancellationToken);
            Assert.Equal(expected.Quantiles, actual.Quantiles);
            Assert.Equal(expected.Mean, actual.Mean);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
