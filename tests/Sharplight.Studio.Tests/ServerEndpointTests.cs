using System.Text;
using System.Text.Json;
using Sharplight.Audio;
using Sharplight.Hosting.Loaders;
using Sharplight.Imaging;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Server;
using Sharplight.Studio.Tests.Fakes;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The embedded server over a model host holding speech, image and embedding models (what <c>EngineModelRuntime</c> registers): the
/// endpoints of every kind are reachable by model name, also for models registered after the server started, and the served switch applies.
/// </summary>
public sealed class ServerEndpointTests : IAsyncLifetime
{
    private readonly ModelHost _host = new();
    private readonly KestrelApiServer _server;
    private readonly HttpClient _client = new();
    private readonly FakeRecognizer _recognizer = new();
    private readonly FakeSynthesizer _synthesizer = new();
    private readonly FakeImageGenerator _generator = new();
    private readonly FakeTextEmbedder _embedder = new();
    private string _url = "";

    public ServerEndpointTests()
    {
        _server = new KestrelApiServer(_host);
    }

    public async ValueTask InitializeAsync()
    {
        await _server.StartAsync(new ServerSettings { Host = "127.0.0.1", Port = 0 }, TestContext.Current.CancellationToken);
        _url = _server.BaseUrl!;
    }

    [Fact]
    public async Task PortZero_BindsAFreePortAndReportsTheActualOne()
    {
        var uri = new Uri(_url);
        Assert.NotEqual(0, uri.Port);
        using var response = await _client.GetAsync(_url + "/v1/models", TestContext.Current.CancellationToken);
        Assert.True(response.IsSuccessStatusCode);

        await using var second = new KestrelApiServer(_host);
        await second.StartAsync(new ServerSettings { Host = "127.0.0.1", Port = 0 }, TestContext.Current.CancellationToken);
        Assert.NotEqual(uri.Port, new Uri(second.BaseUrl!).Port);
    }

    public async ValueTask DisposeAsync()
    {
        _client.Dispose();
        await _server.DisposeAsync();
        _host.Dispose();
    }

    private void AddAll()
    {
        HostedTaskModel.Register(_host, "whisper", ModelKind.Task, () => new HostedTaskModel(_recognizer));
        HostedTaskModel.Register(_host, "tts", ModelKind.Task, () => new HostedTaskModel(_synthesizer));
        HostedTaskModel.Register(_host, "sd", ModelKind.Task, () => new HostedTaskModel(_generator));
        HostedTaskModel.Register(_host, "embed", ModelKind.Embedding, () => new EmbeddingModelInstance(_embedder, 0, ownsEmbedder: false));
    }

    private async Task<HttpResponseMessage> PostJson(string path, string json)
        => await _client.PostAsync(_url + path, new StringContent(json, Encoding.UTF8, "application/json"), TestContext.Current.CancellationToken);

    [Fact]
    public async Task EveryResidentKindAnswersOnItsEndpoint()
    {
        AddAll();

        var wav = new MemoryStream();
        WavFile.Write(wav, new AudioBuffer(new float[16000], 16000));
        using var form = new MultipartFormDataContent { { new ByteArrayContent(wav.ToArray()), "file", "talk.wav" }, { new StringContent("whisper"), "model" } };
        using var transcription = await _client.PostAsync(_url + "/v1/audio/transcriptions", form, TestContext.Current.CancellationToken);
        Assert.Equal(HttpStatusCode.OK, transcription.StatusCode);
        using var transcript = JsonDocument.Parse(await transcription.Content.ReadAsStringAsync(TestContext.Current.CancellationToken));
        Assert.StartsWith("And so, my fellow Americans", transcript.RootElement.GetProperty("text").GetString(), StringComparison.Ordinal);

        using var speech = await PostJson("/v1/audio/speech", """{"model":"tts","input":"hello brave new world"}""");
        Assert.Equal(HttpStatusCode.OK, speech.StatusCode);
        var spoken = WavFile.Read(new MemoryStream(await speech.Content.ReadAsByteArrayAsync(TestContext.Current.CancellationToken)));
        Assert.Equal(24000, spoken.SampleRate);
        Assert.Equal("hello brave new world", _synthesizer.Requests.Single().Text);

        using var image = await PostJson("/v1/images/generations", """{"model":"sd","prompt":"a lighthouse","size":"32x16"}""");
        Assert.Equal(HttpStatusCode.OK, image.StatusCode);
        using var picture = JsonDocument.Parse(await image.Content.ReadAsStringAsync(TestContext.Current.CancellationToken));
        var png = PngCodec.Decode(Convert.FromBase64String(picture.RootElement.GetProperty("data")[0].GetProperty("b64_json").GetString()!));
        Assert.Equal((32, 16), (png.Width, png.Height));

        using var embeddings = await PostJson("/v1/embeddings", """{"model":"embed","input":["a cat","a ship"]}""");
        Assert.Equal(HttpStatusCode.OK, embeddings.StatusCode);
        using var vectors = JsonDocument.Parse(await embeddings.Content.ReadAsStringAsync(TestContext.Current.CancellationToken));
        Assert.Equal(2, vectors.RootElement.GetProperty("data").GetArrayLength());
        Assert.Equal(5, vectors.RootElement.GetProperty("data")[0].GetProperty("embedding").GetArrayLength());
    }

    [Fact]
    public async Task ModelsLoadedAfterTheServerStartedAreReachable_AndUnloadedOnesAreNot()
    {
        using var before = await PostJson("/v1/images/generations", """{"model":"sd","prompt":"x"}""");
        Assert.Equal(HttpStatusCode.NotFound, before.StatusCode);

        AddAll();
        using var loaded = await PostJson("/v1/images/generations", """{"model":"sd","prompt":"x"}""");
        Assert.Equal(HttpStatusCode.OK, loaded.StatusCode);

        await _host.UnregisterAsync("sd", TimeSpan.FromSeconds(1), force: true);
        using var gone = await PostJson("/v1/images/generations", """{"model":"sd","prompt":"x"}""");
        Assert.Equal(HttpStatusCode.NotFound, gone.StatusCode);
    }

    [Fact]
    public async Task TheServedSwitch_HidesAModelFromEveryEndpoint()
    {
        AddAll();
        _server.SetServedModels(["whisper", "embed"]);
        using var hidden = await PostJson("/v1/audio/speech", """{"model":"tts","input":"hi"}""");
        Assert.Equal(HttpStatusCode.NotFound, hidden.StatusCode);
        using var served = await PostJson("/v1/embeddings", """{"model":"embed","input":"a cat"}""");
        Assert.Equal(HttpStatusCode.OK, served.StatusCode);
        Assert.Empty(_synthesizer.Requests);
    }

    [Fact]
    public async Task AModelOfAnotherKind_IsRefusedByName()
    {
        AddAll();
        using var wrong = await PostJson("/v1/images/generations", """{"model":"tts","prompt":"x"}""");
        Assert.Equal(HttpStatusCode.NotFound, wrong.StatusCode);
        Assert.Contains("image generation", await wrong.Content.ReadAsStringAsync(TestContext.Current.CancellationToken), StringComparison.Ordinal);
    }
}
