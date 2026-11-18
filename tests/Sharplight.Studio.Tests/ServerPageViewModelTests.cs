using Sharplight.Llm.Serving.Hosting;
using Sharplight.Studio.Services.Server;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Server;

namespace Sharplight.Studio.Tests;

public sealed class ServerPageViewModelTests
{
    [Fact]
    public async Task StartAndStop_FollowTheServer_AndRequestsAreLogged()
    {
        using var services = new TestServices();
        var server = new ServerPageViewModel(services.Services) { Port = 9123, Host = "0.0.0.0" };
        await server.StartCommand.ExecuteAsync(null);

        Assert.True(server.IsRunning);
        Assert.Equal("http://0.0.0.0:9123", server.BaseUrl);
        Assert.Equal(9123, services.Settings.Current.Server.Port);

        services.Server.Log(new HttpLogEntry(DateTimeOffset.UtcNow, "POST", "/v1/messages", 200, 12, "127.0.0.1"));
        Assert.Equal(1, server.RequestCount);
        Assert.True(Assert.Single(server.Log).IsSuccess);

        await server.StopCommand.ExecuteAsync(null);
        Assert.False(server.IsRunning);
        Assert.Equal("Stopped", server.StatusText);
    }

    [Fact]
    public async Task ARunningServer_IsShownAsRunning()
    {
        using var services = new TestServices();
        await services.Server.StartAsync(new Models.ServerSettings(), CancellationToken.None);
        Assert.True(new ServerPageViewModel(services.Services).IsRunning);
    }

    [Fact]
    public void ServedSwitches_ApplyToTheRunningServer_AndFollowLoads()
    {
        using var services = new TestServices();
        services.Runtime.Add(new FakeModel("a"));
        services.Runtime.Add(new FakeModel("b"));
        var server = new ServerPageViewModel(services.Services);
        Assert.Equal(2, server.Models.Count);
        Assert.All(server.Models, model => Assert.True(model.IsServed));

        server.Models[1].IsServed = false;
        Assert.Equal(["a"], services.Settings.Current.Server.ServedModels);
        Assert.Equal(["a"], services.Server.Served);
        Assert.Contains("\"model\": \"a\"", server.CurlExample);

        services.Runtime.Add(new FakeModel("c"));
        Assert.Equal(3, server.Models.Count);
        Assert.False(server.Models.Single(model => model.Id == "c").IsServed);   // an explicit selection stays explicit

        server.Models[1].IsServed = true;
        server.Models[2].IsServed = true;
        Assert.Empty(services.Settings.Current.Server.ServedModels);   // all served = no filter
    }

    [Fact]
    public void KestrelFilter_IsLive()
    {
        using var host = new ModelHost();
        var kestrel = new KestrelApiServer(host);
        Assert.True(kestrel.IsServed("anything"));
        kestrel.SetServedModels(["a"]);
        Assert.True(kestrel.IsServed("a"));
        Assert.False(kestrel.IsServed("b"));
        kestrel.SetServedModels([]);
        Assert.True(kestrel.IsServed("b"));
    }

    [Fact]
    public void TheEndpointList_ShowsWhichModelsAreReachableOnEach()
    {
        using var services = new TestServices();
        services.Runtime.Add(new FakeModel("qwen"));
        services.Runtime.Add(new FakeModel("whisper") { Kind = Sharplight.Models.ModelKinds.SpeechRecognition });
        services.Runtime.Add(new FakeModel("kokoro") { Kind = Sharplight.Models.ModelKinds.SpeechSynthesis });
        services.Runtime.Add(new FakeModel("sdxl") { Kind = Sharplight.Models.ModelKinds.ImageGeneration });
        services.Runtime.Add(new FakeModel("bge") { Kind = Models.ModelKindCatalog.Embeddings });
        services.Runtime.Add(new FakeModel("depth") { Kind = Sharplight.Models.ModelKinds.DepthEstimation });
        var server = new ServerPageViewModel(services.Services);

        List<string>? At(string path) => server.Endpoints.Single(endpoint => endpoint.Path == path).Models?.ToList();
        Assert.Equal(["qwen"], At("/v1/chat/completions"));
        Assert.Equal(["qwen"], At("/v1/messages"));
        Assert.Equal(["whisper"], At("/v1/audio/transcriptions"));
        Assert.Equal(["whisper"], At("/v1/audio/translations"));
        Assert.Equal(["kokoro"], At("/v1/audio/speech"));
        Assert.Equal(["sdxl"], At("/v1/images/generations"));
        Assert.Equal(["bge"], At("/v1/embeddings"));
        Assert.Null(At("/health"));
        Assert.Equal("no resident model", new ServerPageViewModel(new TestServices().Services).Endpoints.Single(endpoint => endpoint.Path == "/v1/embeddings").ModelsText);

        // Per model, and switching one off or turning an API off follows.
        Assert.Equal("/v1/audio/speech", server.Models.Single(model => model.Id == "kokoro").EndpointsText);
        Assert.Equal("not served over HTTP", server.Models.Single(model => model.Id == "depth").EndpointsText);
        server.Models.Single(model => model.Id == "sdxl").IsServed = false;
        Assert.Empty(At("/v1/images/generations")!);
        server.Anthropic = false;
        Assert.DoesNotContain(server.Endpoints, endpoint => endpoint.Path == "/v1/messages");
        Assert.Equal("/v1/chat/completions  /v1/completions", server.Models.Single(model => model.Id == "qwen").EndpointsText);
    }
}
