using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.Logging;
using Sharplight.AspNetCore;

namespace Sharplight.Serving.Tests;

/// <summary><c>/v1/timeseries/forecast</c> over fake forecasters: request parsing, batching, joint requests with covariates, errors.</summary>
public sealed class TimeSeriesApiTests : IAsyncLifetime
{
    private readonly FakeForecaster _bolt = new();
    private readonly FakeForecaster _other = new();
    private WebApplication _app = null!;
    private HttpClient _client = null!;

    public async ValueTask InitializeAsync()
    {
        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseTestServer();
        builder.Logging.ClearProviders();
        builder.Services.AddSharplightTimeSeriesForecaster(_bolt, "bolt", models => models.MaxHorizon = 100);
        builder.Services.AddSharplightTimeSeriesForecaster(_other, "other");
        _app = builder.Build();
        _app.MapSharplightTimeSeries();
        await _app.StartAsync();
        _client = _app.GetTestClient();
    }

    public async ValueTask DisposeAsync()
    {
        _client.Dispose();
        await _app.StopAsync();
        await _app.DisposeAsync();
    }

    private async Task<(HttpStatusCode Status, JsonElement Body)> Post(string json)
    {
        using var response = await _client.PostAsync("/v1/timeseries/forecast", new StringContent(json, Encoding.UTF8, "application/json"),
            TestContext.Current.CancellationToken);
        string text = await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken);
        return (response.StatusCode, JsonDocument.Parse(text).RootElement.Clone());
    }

    [Fact]
    public async Task ForecastsSeveralSeriesInOneBatchWithTheRequestedOptions()
    {
        var (status, body) = await Post("""
            {"model":"bolt","series":[[1,2,null,4],[null,7,8]],"horizon":3,"quantile_levels":[0.9,0.1],"samples":30,"temperature":0.5,"top_k":10,
             "seed":42,"patch_size":16}
            """);
        Assert.Equal(HttpStatusCode.OK, status);
        var (histories, options) = Assert.Single(_bolt.Batches);
        Assert.Equal(2, histories.Length);
        Assert.True(float.IsNaN(histories[0][2]));
        Assert.True(float.IsNaN(histories[1][0]));
        Assert.Equal(8f, histories[1][2]);
        Assert.Equal(3, options!.Horizon);
        Assert.Equal([0.9f, 0.1f], options.QuantileLevels);
        Assert.Equal(30, options.SampleCount);
        Assert.Equal(0.5f, options.Temperature);
        Assert.Equal(10, options.TopK);
        Assert.Equal(42, options.Seed);
        Assert.Equal(16, options.PatchSize);

        Assert.Equal("timeseries.forecast", body.GetProperty("object").GetString());
        Assert.Equal("bolt", body.GetProperty("model").GetString());
        Assert.Equal(3, body.GetProperty("horizon").GetInt32());
        Assert.Equal(2, body.GetProperty("quantile_levels").GetArrayLength());
        var forecasts = body.GetProperty("forecasts");
        Assert.Equal(2, forecasts.GetArrayLength());
        Assert.Equal(1, forecasts[1].GetProperty("index").GetInt32());
        Assert.Equal([8f, 8f, 8f], forecasts[1].GetProperty("mean").EnumerateArray().Select(v => v.GetSingle()));
        Assert.Equal(4f + 0.9f, forecasts[0].GetProperty("quantiles")[0][2].GetSingle());
        Assert.Equal(5, body.GetProperty("usage").GetProperty("observations").GetInt64());
        Assert.Equal(2, body.GetProperty("usage").GetProperty("series").GetInt32());
    }

    [Fact]
    public async Task AFlatArrayIsOneSeriesAndTheModelsDefaultsApply()
    {
        var (status, body) = await Post("""{"model":"bolt","series":[3, 5]}""");
        Assert.Equal(HttpStatusCode.OK, status);
        var (histories, options) = Assert.Single(_bolt.Batches);
        Assert.Equal([3f, 5f], Assert.Single(histories));
        Assert.Null(options!.Horizon);
        Assert.Null(options.QuantileLevels);
        Assert.Equal(4, body.GetProperty("horizon").GetInt32());
        Assert.Equal([0.1f, 0.5f, 0.9f], body.GetProperty("quantile_levels").EnumerateArray().Select(v => v.GetSingle()));
    }

    [Fact]
    public async Task CovariatesMakeAJointRequest()
    {
        var (status, body) = await Post("""
            {"model":"other","series":[[1,2,3],[4,5,6]],"horizon":2,
             "covariates":[{"name":"price","history":[1,1,null],"future":[2,2]},{"history":[0,1,0]}]}
            """);
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Empty(_other.Batches);
        var request = Assert.Single(_other.JointRequests);
        Assert.Equal(2, request.Targets.Count);
        Assert.Equal("price", request.Covariates[0].Name);
        Assert.Equal([2f, 2f], request.Covariates[0].Future.ToArray());
        Assert.True(request.Covariates[1].Future.IsEmpty);
        Assert.Equal(2, body.GetProperty("forecasts").GetArrayLength());
        Assert.Equal(4, body.GetProperty("usage").GetProperty("series").GetInt32());

        (status, _) = await Post("""{"model":"other","series":[[1,2],[3,4]],"joint":true}""");
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Equal(2, _other.JointRequests.Count);
    }

    [Theory]
    [InlineData("""{"model":"bolt"}""", "series")]
    [InlineData("""{"model":"bolt","series":[]}""", "series")]
    [InlineData("""{"model":"bolt","series":[["a"]]}""", "series")]
    [InlineData("""{"model":"bolt","series":[[1],2]}""", "series")]
    [InlineData("""{"model":"bolt","series":[1],"horizon":0}""", "horizon")]
    [InlineData("""{"model":"bolt","series":[1],"horizon":101}""", "horizon")]
    [InlineData("""{"model":"bolt","series":[1],"quantile_levels":[0.5,1]}""", "quantile_levels")]
    [InlineData("""{"model":"bolt","series":[1],"samples":0}""", "samples")]
    [InlineData("""{"model":"bolt","series":[1],"temperature":0}""", "temperature")]
    [InlineData("""{"model":"bolt","series":[1],"seed":1e12}""", "seed")]
    [InlineData("""{"model":"bolt","series":[1],"covariates":[{"name":"x"}]}""", "covariates")]
    [InlineData("""{"model":"bolt","series":[1],"patch_size":"best"}""", "patch_size")]
    public async Task InvalidRequestsAreRejectedWithTheirField(string json, string param)
    {
        var (status, body) = await Post(json);
        Assert.Equal(HttpStatusCode.BadRequest, status);
        Assert.Equal(param, body.GetProperty("error").GetProperty("param").GetString());
        Assert.Empty(_bolt.Batches);
    }

    [Fact]
    public async Task AutoPatchSizeIsPassedOn()
    {
        var (status, _) = await Post("""{"model":"bolt","series":[1,2,3],"patch_size":"auto"}""");
        Assert.Equal(HttpStatusCode.OK, status);
        var (_, options) = Assert.Single(_bolt.Batches);
        Assert.Equal(Sharplight.TimeSeries.TimeSeriesForecastOptions.AutoPatchSize, options!.PatchSize);
    }

    [Fact]
    public async Task ASeriesWithoutObservationsIsABadRequest()
    {
        var (status, body) = await Post("""{"model":"bolt","series":[[1,2],[null,null]]}""");
        Assert.Equal(HttpStatusCode.BadRequest, status);
        Assert.Contains("observed", body.GetProperty("error").GetProperty("message").GetString(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task AnUnknownModelIsNotFoundWhileSeveralAreRegistered()
    {
        var (status, body) = await Post("""{"model":"nope","series":[1,2]}""");
        Assert.Equal(HttpStatusCode.NotFound, status);
        Assert.Equal("model_not_found", body.GetProperty("error").GetProperty("code").GetString());

        // Without a name the default (the first registered) answers.
        (status, body) = await Post("""{"series":[1,2]}""");
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Equal("bolt", body.GetProperty("model").GetString());
    }
}
