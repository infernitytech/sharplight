namespace Sharplight.Hub;

/// <summary>Connection settings shared by <see cref="HubClient"/> and <see cref="FileDownloader"/>.</summary>
public sealed class HubClientOptions
{
    /// <summary>Environment variable read by <see cref="FromEnvironment"/> for the access token.</summary>
    public const string TokenVariable = "HF_TOKEN";

    /// <summary>Environment variable read by <see cref="FromEnvironment"/> for another Hub endpoint (a mirror).</summary>
    public const string EndpointVariable = "HF_ENDPOINT";

    /// <summary>The public Hugging Face Hub.</summary>
    public static readonly Uri DefaultEndpoint = new("https://huggingface.co");

    /// <summary>Base address of the Hub.</summary>
    public Uri Endpoint { get; init; } = DefaultEndpoint;

    /// <summary>
    /// Access token for private and gated repositories, sent as a bearer token to <see cref="Endpoint"/> only (redirects to the
    /// download CDN drop it). Never logged.
    /// </summary>
    public string? Token { get; init; }

    /// <summary>User-Agent sent with every request; generic, without user or machine data.</summary>
    public string UserAgent { get; init; } = "sharplight/0.1";

    /// <summary>Timeout of a metadata request (search, file listing, small files).</summary>
    public TimeSpan RequestTimeout { get; init; } = TimeSpan.FromSeconds(30);

    /// <summary>A download attempt that receives no bytes for this long is abandoned and retried.</summary>
    public TimeSpan StallTimeout { get; init; } = TimeSpan.FromSeconds(60);

    /// <summary>Retry behaviour for transient failures.</summary>
    public RetryOptions Retry { get; init; } = new();

    /// <summary>Options with the token from <c>HF_TOKEN</c> and the endpoint from <c>HF_ENDPOINT</c> when they are set.</summary>
    public static HubClientOptions FromEnvironment()
    {
        var token = Environment.GetEnvironmentVariable(TokenVariable);
        var endpoint = Environment.GetEnvironmentVariable(EndpointVariable);
        return new HubClientOptions
        {
            Token = string.IsNullOrWhiteSpace(token) ? null : token.Trim(),
            Endpoint = Uri.TryCreate(endpoint, UriKind.Absolute, out var uri) ? uri : DefaultEndpoint,
        };
    }

    /// <summary>
    /// Creates an <see cref="HttpClient"/> suited to large downloads: no overall timeout (the stall and request timeouts
    /// apply instead), no transparent decompression (byte ranges must address the stored bytes), redirects followed.
    /// </summary>
    public static HttpClient CreateHttpClient()
    {
        var handler = new SocketsHttpHandler
        {
            AutomaticDecompression = System.Net.DecompressionMethods.None,
            ConnectTimeout = TimeSpan.FromSeconds(30),
            PooledConnectionLifetime = TimeSpan.FromMinutes(10),
            AllowAutoRedirect = true,
        };
        return new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
    }
}
