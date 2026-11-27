using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Sharplight.AspNetCore;
using Sharplight.Hosting;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Models;

namespace Sharplight.Serving.Tests;

/// <summary>
/// <c>/v1/timeseries/forecast</c> over a real Chronos-Bolt checkpoint (<c>~/models/timeseries/chronos-bolt-tiny</c>, skipped without it) hosted as a task model the
/// way <c>sharplight serve --model bolt=PATH,task=time-series</c> does: Sharplight.Hosting on the CPU, the server serving the host's models.
/// </summary>
[Collection("RealModel")]
public sealed class TimeSeriesRealTests
{
    private static readonly string s_model = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "timeseries", "chronos-bolt-tiny");

    [Fact]
    public async Task ForecastsABatchOfSeriesWithChronosBolt()
    {
        Assert.SkipUnless(Directory.Exists(s_model), $"{s_model} not found");
        var ct = TestContext.Current.CancellationToken;
        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseTestServer();
        builder.Logging.ClearProviders();
        builder.Services.AddSharplight();
        builder.Services.AddSharplightHosting(options =>
        {
            options.Reserve = 0;
            options.Defaults.Device = DeviceSpecifications.Cpu;
            options.Models.Add(new HostedModelOptions { Name = "bolt", Path = s_model, Kind = ModelKind.Task, Task = ModelKinds.TimeSeriesForecasting });
        });
        builder.Services.AddSharplightServer(options => options.ServeHostModels = true);
        await using var app = builder.Build();
        app.UseSharplightServer();
        app.MapSharplightServer();
        await app.StartAsync(ct);
        using var client = app.GetTestClient();

        // A sine wave, the same wave scaled by 10 and shifted by 100, and a ramp with gaps: Chronos-Bolt's instance scaling makes the second forecast
        // the first one scaled and shifted.
        var sine = Enumerable.Range(0, 200).Select(i => MathF.Sin(i * 2 * MathF.PI / 24)).ToArray();
        var series = new[]
        {
            sine.Select(v => (float?)v).ToArray(),
            sine.Select(v => (float?)((10 * v) + 100)).ToArray(),
            Enumerable.Range(0, 100).Select(i => i % 10 == 3 ? null : (float?)i).ToArray(),
        };
        string json = JsonSerializer.Serialize(new { model = "bolt", series, horizon = 80, quantile_levels = new[] { 0.1, 0.5, 0.9 } });
        using var response = await client.PostAsync("/v1/timeseries/forecast", new StringContent(json, Encoding.UTF8, "application/json"), ct);
        string text = await response.Content.ReadAsStringAsync(ct);
        Assert.True(response.StatusCode == HttpStatusCode.OK, text);
        using var body = JsonDocument.Parse(text);
        var forecasts = body.RootElement.GetProperty("forecasts");
        Assert.Equal(3, forecasts.GetArrayLength());
        Assert.Equal(80, body.RootElement.GetProperty("horizon").GetInt32());
        float[] Mean(int i) => [.. forecasts[i].GetProperty("mean").EnumerateArray().Select(v => v.GetSingle())];
        var wave = Mean(0);
        var scaled = Mean(1);
        Assert.Equal(80, wave.Length);
        for (int t = 0; t < wave.Length; t++)
        {
            Assert.True(Math.Abs(scaled[t] - ((10 * wave[t]) + 100)) < 1e-2, $"step {t}: {scaled[t]} vs {(10 * wave[t]) + 100}");
        }

        // The wave continues (step 200 + t is a trough at t = 10, a peak at t = 22), the ramp keeps rising, and the quantiles are ordered.
        Assert.True(wave[22] > 0.5f && wave[10] < -0.5f, $"sine forecast {wave[22]}, {wave[10]}");
        var ramp = Mean(2);
        Assert.True(ramp[0] > 95 && ramp[^1] > ramp[0], $"ramp forecast {ramp[0]} .. {ramp[^1]}");
        var quantiles = forecasts[2].GetProperty("quantiles");
        Assert.True(quantiles[0][10].GetSingle() <= quantiles[1][10].GetSingle() && quantiles[1][10].GetSingle() <= quantiles[2][10].GetSingle());
        await app.StopAsync(ct);
    }
}
