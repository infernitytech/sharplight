using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Sharplight.AspNetCore;
using Sharplight.ImageGeneration;
using Sharplight.Imaging;

namespace Sharplight.Serving.Tests;

/// <summary><c>/v1/images/generations</c> over a fake generator.</summary>
public sealed class ImageApiTests : IAsyncLifetime
{
    private readonly FakeImageGenerator _generator = new();
    private WebApplication _app = null!;
    private HttpClient _client = null!;

    public async ValueTask InitializeAsync()
    {
        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseTestServer();
        builder.Logging.ClearProviders();
        builder.Services.AddSingleton<IImageGenerator>(_generator);
        builder.Services.AddSharplightImageGenerator(name: "sd15");
        _app = builder.Build();
        _app.MapSharplightImages();
        await _app.StartAsync();
        _client = _app.GetTestClient();
    }

    public async ValueTask DisposeAsync()
    {
        await _app.StopAsync();
        await _app.DisposeAsync();
    }

    private async Task<(HttpStatusCode Status, JsonElement Body)> Post(string json)
    {
        using var response = await _client.PostAsync("/v1/images/generations", new StringContent(json, Encoding.UTF8, "application/json"),
            TestContext.Current.CancellationToken);
        string text = await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken);
        return (response.StatusCode, JsonDocument.Parse(text).RootElement.Clone());
    }

    [Fact]
    public async Task GeneratesPngImagesWithTheRequestedOptions()
    {
        var (status, body) = await Post("""
            {"model":"sd15","prompt":"a lighthouse","negative_prompt":"blurry","n":2,"size":"16x8","steps":3,
             "guidance_scale":5.5,"seed":42,"sampler":"euler_a","response_format":"b64_json"}
            """);
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Equal(42, body.GetProperty("seed").GetInt64());
        var data = body.GetProperty("data");
        Assert.Equal(2, data.GetArrayLength());
        var second = PngCodec.Decode(Convert.FromBase64String(data[1].GetProperty("b64_json").GetString()!));
        Assert.Equal(16, second.Width);
        Assert.Equal(8, second.Height);
        Assert.Equal(43, second.Rgb[0]);

        var (request, options) = Assert.Single(_generator.Calls);
        Assert.Equal("a lighthouse", request.Prompt);
        Assert.Equal("blurry", request.NegativePrompt);
        Assert.Equal(3, options.Steps);
        Assert.Equal(5.5f, options.GuidanceScale);
        Assert.Equal("euler_a", options.Scheduler);
    }

    [Fact]
    public async Task SingleModelAnswersToAnyName()
    {
        var (status, body) = await Post("""{"model":"dall-e-2","prompt":"a cube"}""");
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Equal(1, body.GetProperty("data").GetArrayLength());
    }

    [Theory]
    [InlineData("""{"model":"sd15"}""", "prompt")]
    [InlineData("""{"prompt":"x","size":"big"}""", "size")]
    [InlineData("""{"prompt":"x","n":9}""", "n")]
    [InlineData("""{"prompt":"x","response_format":"url"}""", "response_format")]
    public async Task RejectsInvalidRequests(string json, string param)
    {
        var (status, body) = await Post(json);
        Assert.Equal(HttpStatusCode.BadRequest, status);
        Assert.Equal(param, body.GetProperty("error").GetProperty("param").GetString());
    }
}
