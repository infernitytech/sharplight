using Sharplight.Hub;

namespace Sharplight.Server;

/// <summary>The Hub client, downloader, model store and puller one command uses, on one HTTP client (token from <c>HF_TOKEN</c>).</summary>
internal sealed class HubContext : IDisposable
{
    private readonly HttpClient _http;

    public HubContext(string storeRoot)
    {
        var options = HubClientOptions.FromEnvironment();
        _http = HubClientOptions.CreateHttpClient();
        Hub = new HubClient(_http, options);
        Store = new ModelStore(storeRoot);
        Puller = new ModelPuller(Hub, new FileDownloader(_http, options), Store);
    }

    public HubClient Hub { get; }

    public ModelStore Store { get; }

    public ModelPuller Puller { get; }

    public void Dispose() => _http.Dispose();
}
