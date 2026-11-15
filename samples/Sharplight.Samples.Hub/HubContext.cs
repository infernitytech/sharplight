using Sharplight.Hub;

namespace Sharplight.Samples.Hub;

/// <summary>The Hub client, downloader, model store and puller one command uses, on one HTTP client (token from <c>HF_TOKEN</c>).</summary>
public sealed class HubContext : IDisposable
{
    private readonly HttpClient _http;

    /// <summary>Creates the clients; pulled models are converted with the runtime's converter.</summary>
    /// <param name="runtime">The Sharplight services.</param>
    /// <param name="storeRoot">The model store directory.</param>
    public HubContext(SampleRuntime runtime, string storeRoot)
    {
        var options = HubClientOptions.FromEnvironment();
        _http = HubClientOptions.CreateHttpClient();
        Hub = new HubClient(_http, options);
        Store = new ModelStore(storeRoot);
        Puller = new ModelPuller(Hub, new FileDownloader(_http, options), Store, runtime.Converter);
    }

    public HubClient Hub { get; }

    public ModelStore Store { get; }

    public ModelPuller Puller { get; }

    public void Dispose() => _http.Dispose();
}
