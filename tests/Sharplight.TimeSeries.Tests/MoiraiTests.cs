using System.Text.Json;

namespace Sharplight.TimeSeries.Tests;

/// <summary>
/// Moirai against uni2ts on the same weights (<c>tools/timeseries-reference/reference_moirai.py</c>): Moirai 2.0's quantile forecasts exactly (Moirai2Forecast), Moirai
/// 1.1's mixture parameters per step exactly and its sampled quantiles statistically (MoiraiForecast at patch size 32; torch's RNG is not reproduced).
/// </summary>
public class MoiraiTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Moirai2_forecasts_match_the_reference(bool gpu)
    {
        const string name = "moirai-2.0-R-small";
        TimeSeriesModels.Require(TimeSeriesModels.Model(name), TimeSeriesModels.Reference(name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu), TimeSeriesModels.Model(name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        Assert.Equal(1680, forecaster.ContextLength);
        Assert.Equal(64, forecaster.PredictionLength);
        Assert.Equal([0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f], forecaster.QuantileLevels);
        const double bound = 1e-4;
        foreach (var expected in TimeSeriesModels.Cases(name))
        {
            var forecast = await forecaster.ForecastAsync(expected.History, new TimeSeriesForecastOptions { Horizon = expected.Horizon, QuantileLevels = expected.Levels },
                TestContext.Current.CancellationToken);
            Assert.Equal(expected.Levels, forecast.QuantileLevels);
            Assert.Equal(expected.Horizon, forecast.Horizon);
            double error = TimeSeriesModels.RelativeError(forecast.Quantiles, expected.Quantiles);
            double meanError = TimeSeriesModels.RelativeError([forecast.Mean], [expected.Mean]);
            TestContext.Current.TestOutputHelper?.WriteLine($"{expected.Name}: quantile error {error:E2}, point error {meanError:E2} (of the forecast's range)");
            Assert.True(error < bound, $"{expected.Name}: quantile error {error:E2}");
            Assert.True(meanError < bound, $"{expected.Name}: point error {meanError:E2}");
        }
    }

    [Theory]
    [InlineData("moirai-1.1-R-small", false)]
    [InlineData("moirai-1.1-R-small", true)]
    [InlineData("moirai-1.1-R-base", false)]
    [InlineData("moirai-1.1-R-large", false)]
    public async Task Moirai1_distribution_matches_the_reference(string name, bool gpu)
    {
        TimeSeriesModels.Require(TimeSeriesModels.Model(name), TimeSeriesModels.Reference(name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu), TimeSeriesModels.Model(name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        Assert.Equal(1000, forecaster.ContextLength);
        var client = Assert.IsType<MaskedEncoderForecasterClient>(forecaster);
        Assert.Equal([8, 16, 32, 64, 128], client.PatchSizes);
        using var document = JsonDocument.Parse(File.ReadAllText(TimeSeriesModels.Reference(name)));
        var cases = TimeSeriesModels.Cases(name);
        int index = 0;
        foreach (var c in document.RootElement.GetProperty("cases").EnumerateArray())
        {
            var expected = cases[index++];
            var (loc, scale, parameters) = await client.DistributionAsync(expected.History, expected.Horizon, c.GetProperty("patch_size").GetInt32());
            Assert.Equal(c.GetProperty("loc").GetDouble(), loc, 1e-5 * Math.Max(1, Math.Abs(loc)));
            Assert.Equal(c.GetProperty("scale").GetDouble(), scale, 1e-5 * Math.Max(1, scale));
            var p = c.GetProperty("params");
            var reference = new List<(string Name, double[] Values, int Column)>
            {
                ("student_t.df", Values(p, "student_t", "df"), 4), ("student_t.loc", Values(p, "student_t", "loc"), 5),
                ("student_t.scale", Values(p, "student_t", "scale"), 6), ("normal.loc", Values(p, "normal", "loc"), 7),
                ("negative_binomial.total_count", Values(p, "negative_binomial", "total_count"), 8),
                ("negative_binomial.logits", Values(p, "negative_binomial", "logits"), 9), ("log_normal.loc", Values(p, "log_normal", "loc"), 10),
                ("log_normal.scale", Values(p, "log_normal", "scale"), 11),
            };
            var weights = p.GetProperty("weights").EnumerateArray().Select(row => row.EnumerateArray().Select(v => v.GetDouble()).ToArray()).ToArray();
            for (int k = 0; k < 4; k++)
            {
                reference.Add(($"weights[{k}]", [.. weights.Select(row => row[k])], k));
            }

            double worst = 0;
            string worstName = "";
            foreach (var (parameter, values, column) in reference)
            {
                for (int t = 0; t < expected.Horizon; t++)
                {
                    double error = Math.Abs(parameters[t][column] - values[t]) / Math.Max(1, Math.Abs(values[t]));
                    if (error > worst)
                    {
                        worst = error;
                        worstName = $"{parameter} at step {t}";
                    }
                }
            }

            TestContext.Current.TestOutputHelper?.WriteLine($"{expected.Name}: largest parameter error {worst:E2} ({worstName})");
            Assert.True(worst < 1e-3, $"{expected.Name}: parameter error {worst:E2} ({worstName})");
        }
    }

    // In probability space, so heavy tails do not matter: the reference's empirical quantile at level l (20000 torch samples) lies between this forecast's
    // quantiles at l ± 0.025 (20000 seeded samples; about five standard errors of the difference of two sample quantiles' levels, sqrt(2 · 0.25 / 20000)).
    [Theory]
    [InlineData("moirai-1.1-R-small")]
    public async Task Moirai1_sampled_quantiles_match_the_reference_statistically(string name)
    {
        TimeSeriesModels.Require(TimeSeriesModels.Model(name), TimeSeriesModels.Reference(name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu: false), TimeSeriesModels.Model(name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        Assert.Equal(64, forecaster.PredictionLength);
        const float delta = 0.025f;
        foreach (var expected in TimeSeriesModels.Cases(name))
        {
            float[] bracket = [.. expected.Levels.SelectMany(l => new[] { l - delta, l + delta })];
            var forecast = await forecaster.ForecastAsync(expected.History,
                new TimeSeriesForecastOptions { Horizon = expected.Horizon, QuantileLevels = bracket, SampleCount = 20000, Seed = 5 },
                TestContext.Current.CancellationToken);
            int outside = 0;
            for (int l = 0; l < expected.Levels.Length; l++)
            {
                var low = forecast.Quantiles[2 * l];
                var high = forecast.Quantiles[(2 * l) + 1];
                for (int t = 0; t < expected.Horizon; t++)
                {
                    float value = expected.Quantiles[l][t];
                    float slack = 1e-5f * Math.Max(1f, Math.Abs(value));
                    if (value < low[t] - slack || value > high[t] + slack)
                    {
                        outside++;
                        TestContext.Current.TestOutputHelper?.WriteLine($"{expected.Name}: level {expected.Levels[l]} step {t}: {value} not in [{low[t]}, {high[t]}]");
                    }
                }
            }

            TestContext.Current.TestOutputHelper?.WriteLine($"{expected.Name}: {outside} of {expected.Levels.Length * expected.Horizon} reference quantiles outside the bracket");
            Assert.Equal(0, outside);
        }
    }

    [Fact]
    public async Task Moirai1_seeded_forecasts_repeat_and_patch_sizes_are_checked()
    {
        const string name = "moirai-1.1-R-small";
        TimeSeriesModels.Require(TimeSeriesModels.Model(name), TimeSeriesModels.Reference(name));
        using var model = TimeSeriesModels.Load(TimeSeriesModels.Open(gpu: false), TimeSeriesModels.Model(name));
        using var forecaster = TimeSeriesModels.Client<ITimeSeriesForecaster>(model);
        var history = TimeSeriesModels.Cases(name)[0].History;
        var options = new TimeSeriesForecastOptions { Horizon = 40, SampleCount = 50, Seed = 3, PatchSize = 16 };
        var a = await forecaster.ForecastAsync(history, options, TestContext.Current.CancellationToken);
        var b = await forecaster.ForecastAsync(history, options, TestContext.Current.CancellationToken);
        Assert.Equal(a.Quantiles, b.Quantiles);
        Assert.Equal(a.Mean, b.Mean);
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => forecaster.ForecastAsync(history, options with { PatchSize = 12 }, TestContext.Current.CancellationToken));
    }

    [Fact]
    public void Time_ids_and_scaling_follow_uni2ts()
    {
        float n = float.NaN;
        var (values, observed) = MoiraiPatches.Pad([n, n, n, 1, 2, n, 4, 5, 6], 2);   // 10 values: one padding value, then the context
        Assert.Equal([0, 0, 0, 0, 1, 2, 0, 4, 5, 6], values);
        Assert.Equal([0, 0, 0, 1, 2], MoiraiPatches.TimeIds(observed, 2));
        var (loc, scale) = MoiraiPatches.Scale(values, observed);
        Assert.Equal(3.6f, loc, 1e-6f);
        Assert.Equal((float)Math.Sqrt((((2.6 * 2.6) + (1.6 * 1.6) + (0.4 * 0.4) + (1.4 * 1.4) + (2.4 * 2.4)) / 4) + 1e-5), scale, 1e-6f);
        Assert.Equal((float)Math.Sqrt(1e-5), MoiraiPatches.Scale([7f], [true]).Scale, 1e-9f);
    }

    [Fact]
    public void Mixture_samples_follow_the_component_distributions()
    {
        var mixture = new MixtureDistribution("student_t,normal_fixed:0.001,negative_binomial,log_normal");
        var sampler = new VariateSampler(new Random(1));
        // Raw parameters: logits selecting one component at a time, then the components' raw parameters.
        double Mean(int component, float[] raw)
        {
            var logits = new float[4];
            Array.Fill(logits, -50f);
            logits[component] = 50f;
            var mapped = mixture.Map([.. logits, .. raw]);
            double sum = 0;
            for (int i = 0; i < 200000; i++)
            {
                sum += mixture.Sample(mapped, sampler);
            }

            return sum / 200000;
        }

        float[] parameters = [3f, 1.5f, 0f, 2f, 3f, 0.5f, 0.2f, -0.5f];
        double softplus(double x) => Math.Log(1 + Math.Exp(x));
        Assert.Equal(1.5, Mean(0, parameters), 0.02);                                         // Student-t: its loc (df > 2)
        Assert.Equal(2.0, Mean(1, parameters), 0.01);                                         // normal with the fixed scale
        Assert.Equal(softplus(3) * Math.Exp(0.5), Mean(2, parameters), 0.05);                 // negative binomial: r · exp(logits)
        double sigma = softplus(-0.5);
        Assert.Equal(Math.Exp(0.2 + (sigma * sigma / 2)), Mean(3, parameters), 0.02);        // log-normal
        Assert.Equal(25.0, Enumerable.Range(0, 100000).Average(_ => sampler.Poisson(25)), 0.1);
        Assert.Equal(0.3, Enumerable.Range(0, 100000).Average(_ => sampler.Gamma(0.3)), 0.01);
    }

    private static double[] Values(JsonElement p, string component, string parameter)
        => [.. p.GetProperty(component).GetProperty(parameter).EnumerateArray().Select(v => v.GetDouble())];
}
