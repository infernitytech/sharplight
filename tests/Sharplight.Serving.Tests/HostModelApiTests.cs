using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Sharplight.AspNetCore;
using Sharplight.Audio;
using Sharplight.Hosting.Loaders;
using Sharplight.Imaging;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Serving.Tests;

/// <summary>
/// The media and retrieval endpoints over a <see cref="ModelHost"/> (<see cref="SharplightServerOptions.ServeHostModels"/>): models registered
/// after the server started are reachable, a name picks the model, the client leases it for the request, the model filter applies.
/// </summary>
public sealed class HostModelApiTests : IAsyncLifetime
{
    private readonly FakeSpeechRecognizer _recognizer = new();
    private readonly FakeSpeechSynthesizer _synthesizer = new();
    private readonly FakeImageGenerator _generator = new();
    private readonly FakeEmbedder _embedder = new();
    private readonly HashSet<string> _hidden = [];
    private ModelHost _host = null!;
    private WebApplication _app = null!;
    private HttpClient _client = null!;

    public async ValueTask InitializeAsync()
    {
        _host = new ModelHost(new ModelHostOptions { ReserveBytes = 0 });
        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseTestServer();
        builder.Logging.ClearProviders();
        builder.Services.AddSharplightServer(_host, options =>
        {
            options.ServeHostModels = true;
            options.ModelFilter = name => !_hidden.Contains(name);
        });
        _app = builder.Build();
        _app.UseSharplightServer();
        _app.MapSharplightServer();
        await _app.StartAsync();
        _client = _app.GetTestClient();
    }

    public async ValueTask DisposeAsync()
    {
        _client.Dispose();
        await _app.StopAsync();
        await _app.DisposeAsync();
        _host.Dispose();
    }

    private FakeTaskLoader Add(string name, ModelKind kind, Func<LoadedModel> create)
    {
        var loader = new FakeTaskLoader(kind, create);
        _host.Register(new ModelRegistration { Name = name, Loader = loader });
        return loader;
    }

    private async Task<HttpResponseMessage> Json(string path, string json)
        => await _client.PostAsync(path, new StringContent(json, Encoding.UTF8, "application/json"), TestContext.Current.CancellationToken);

    private async Task<HttpResponseMessage> Transcribe(string path, string model)
    {
        var wav = new MemoryStream();
        WavFile.Write(wav, new AudioBuffer(new float[16000], 16000));
        var form = new MultipartFormDataContent
        {
            { new ByteArrayContent(wav.ToArray()), "file", "audio.wav" },
            { new StringContent(model), "model" },
        };
        return await _client.PostAsync(path, form, TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task TranscribesWithAHostModelRegisteredAfterTheServerStarted()
    {
        var loader = Add("whisper", ModelKind.Task, () => new FakeTaskLoaded(_recognizer));
        using var response = await Transcribe("/v1/audio/transcriptions", "whisper");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken));
        Assert.Equal("Hello world.", body.RootElement.GetProperty("text").GetString());
        Assert.Equal(1, loader.Loads);
    }

    [Fact]
    public async Task SpeaksAsWavAndAsStreamedPcm()
    {
        Add("tts", ModelKind.Task, () => new FakeTaskLoaded(_synthesizer));
        using var wav = await Json("/v1/audio/speech", """{"model":"tts","input":"one two three","voice":"echo","instructions":"a calm voice"}""");
        Assert.Equal(HttpStatusCode.OK, wav.StatusCode);
        Assert.Equal("audio/wav", wav.Content.Headers.ContentType!.MediaType);
        var decoded = WavFile.Read(new MemoryStream(await wav.Content.ReadAsByteArrayAsync(TestContext.Current.CancellationToken)));
        Assert.Equal(8000, decoded.SampleRate);
        Assert.Equal(300, decoded.FrameCount);
        Assert.Equal("echo", _synthesizer.LastOptions!.Voice);
        Assert.Equal("a calm voice", _synthesizer.LastOptions.Instructions);

        using var pcm = await Json("/v1/audio/speech", """{"model":"tts","input":"one two","response_format":"pcm"}""");
        var bytes = await pcm.Content.ReadAsByteArrayAsync(TestContext.Current.CancellationToken);
        Assert.Equal(2 * 200, bytes.Length);
        Assert.InRange(BitConverter.ToInt16(bytes, 0), (short)16000, (short)16500);
    }

    [Fact]
    public async Task GeneratesImagesWithAHostModel()
    {
        Add("sd", ModelKind.Task, () => new FakeTaskLoaded(_generator));
        using var response = await Json("/v1/images/generations", """{"model":"sd","prompt":"a lighthouse","size":"16x8","seed":7}""");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken));
        var image = PngCodec.Decode(Convert.FromBase64String(body.RootElement.GetProperty("data")[0].GetProperty("b64_json").GetString()!));
        Assert.Equal((16, 8), (image.Width, image.Height));
        Assert.Equal("a lighthouse", Assert.Single(_generator.Calls).Request.Prompt);
    }

    [Fact]
    public async Task EmbedsWithAHostEmbeddingModel()
    {
        Add("embed", ModelKind.Embedding, () => new TaskClientInstance([_embedder], 0, ModelKind.Embedding, ownsClients: false));
        using var response = await Json("/v1/embeddings", """{"model":"embed","input":["hello","world"]}""");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken));
        Assert.Equal(2, body.RootElement.GetProperty("data").GetArrayLength());
    }

    [Fact]
    public async Task ForecastsWithAHostTimeSeriesModel()
    {
        var forecaster = new FakeForecaster();
        var loader = Add("bolt", ModelKind.Task, () => new FakeTaskLoaded(forecaster));
        using var response = await Json("/v1/timeseries/forecast", """{"model":"bolt","series":[[1,2,3],[4,5,6]],"horizon":2}""");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken));
        Assert.Equal("bolt", body.RootElement.GetProperty("model").GetString());
        Assert.Equal(2, body.RootElement.GetProperty("forecasts").GetArrayLength());
        Assert.Equal(2, Assert.Single(forecaster.Batches).Histories.Length);
        Assert.Equal(1, loader.Loads);

        Add("sd", ModelKind.Task, () => new FakeTaskLoaded(_generator));
        using var wrongKind = await Json("/v1/timeseries/forecast", """{"model":"sd","series":[1,2]}""");
        Assert.Equal(HttpStatusCode.NotFound, wrongKind.StatusCode);
    }

    [Fact]
    public async Task AModelOfAnotherKindIsRefusedWithItsName()
    {
        Add("tts", ModelKind.Task, () => new FakeTaskLoaded(_synthesizer));
        using var response = await Transcribe("/v1/audio/transcriptions", "tts");
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        Assert.Contains("tts", await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken), StringComparison.Ordinal);
    }

    [Fact]
    public async Task UnknownAndFilteredModelsAreNotFound()
    {
        Add("sd", ModelKind.Task, () => new FakeTaskLoaded(_generator));
        using var unknown = await Json("/v1/images/generations", """{"model":"nope","prompt":"x"}""");
        Assert.Equal(HttpStatusCode.NotFound, unknown.StatusCode);
        _hidden.Add("sd");
        using var hidden = await Json("/v1/images/generations", """{"model":"sd","prompt":"x"}""");
        Assert.Equal(HttpStatusCode.NotFound, hidden.StatusCode);
        Assert.Empty(_generator.Calls);
    }

    [Fact]
    public async Task UnloadedModelsAreRemovedFromTheEndpoints()
    {
        Add("sd", ModelKind.Task, () => new FakeTaskLoaded(_generator));
        using var first = await Json("/v1/images/generations", """{"model":"sd","prompt":"x"}""");
        Assert.Equal(HttpStatusCode.OK, first.StatusCode);
        await _host.UnregisterAsync("sd", TimeSpan.FromSeconds(1), force: true);
        using var gone = await Json("/v1/images/generations", """{"model":"sd","prompt":"x"}""");
        Assert.Equal(HttpStatusCode.NotFound, gone.StatusCode);
    }
}
