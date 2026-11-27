using Sharplight.Llm.Serving;

namespace Sharplight.AspNetCore;

/// <summary>
/// Settings of the HTTP server (see <see cref="SharplightServerExtensions"/>): which APIs are exposed, authentication, CORS,
/// request limits and default sampling. Configure through the callback of <c>AddSharplight</c>.
/// </summary>
public sealed class SharplightServerOptions
{
    /// <summary>Expose the OpenAI-compatible API (<c>/v1/chat/completions</c>, <c>/v1/completions</c>, <c>/v1/models</c>).</summary>
    public bool OpenAi { get; set; } = true;
    /// <summary>
    /// Accept the request field <c>cache_salt</c> (vLLM's prefix-cache salt) from clients; it narrows prefix-cache sharing within the
    /// request's scope (<see cref="SharplightRequestContext.CacheSalt"/>, <see cref="SharplightRequestContext.Scope"/>) and can never
    /// widen it. Off by default: a request carrying it is refused with 400, and the cache key is what the host sets.
    /// </summary>
    public bool AllowClientCacheSalt { get; set; }
    /// <summary>Expose the Anthropic-compatible API (<c>/v1/messages</c>, <c>/v1/messages/count_tokens</c>).</summary>
    public bool Anthropic { get; set; } = true;
    /// <summary>Accepted API keys (<c>Authorization: Bearer</c> or <c>x-api-key</c>). Empty = no authentication.</summary>
    public IList<string> ApiKeys { get; set; } = [];
    /// <summary>Allowed CORS origins ("*" for any). Empty = CORS headers are not sent.</summary>
    public IList<string> CorsOrigins { get; set; } = [];
    /// <summary>Largest accepted request body (images arrive base64 encoded in the body).</summary>
    public long MaxRequestBodyBytes { get; set; } = 32 * 1024 * 1024;
    /// <summary>Largest <c>n</c> a request may ask for.</summary>
    public int MaxChoices { get; set; } = 8;
    /// <summary>Total time budget per request (queue + generation); null = none.</summary>
    public TimeSpan? RequestTimeout { get; set; }
    /// <summary>Applied when a request leaves a sampling parameter out.</summary>
    public SamplingParams DefaultSampling { get; set; } = new();
    /// <summary>How long shutdown waits for running requests to finish.</summary>
    public TimeSpan DrainTimeout { get; set; } = TimeSpan.FromSeconds(30);
    /// <summary>Serve <c>/metrics</c> (Prometheus text format).</summary>
    public bool Metrics { get; set; } = true;
    /// <summary>
    /// A request whose <c>model</c> names no registered model is answered by the default (first) chat model instead of a 404.
    /// On for a single-model server, where clients commonly send arbitrary names; off by default with several models.
    /// </summary>
    public bool UnknownModelFallsBackToDefault { get; set; }
    /// <summary>Serve the admin endpoints (<c>GET /admin/models</c>, <c>POST /admin/models/{name}/load|unload</c>). They need an API key when keys are configured, else a loopback caller.</summary>
    public bool Admin { get; set; } = true;
    /// <summary>
    /// Which models of the host the endpoints expose (null = all): models it rejects are not listed and are answered like unknown
    /// names, so one <c>ModelHost</c> can be shared with other code (e.g. a desktop app's chat) while only some models are served.
    /// Evaluated per request. On-demand names (<c>ModelHostOptions.Resolve</c>) are passed through it before they are loaded.
    /// </summary>
    public Func<string, bool>? ModelFilter { get; set; }
    /// <summary>
    /// Drain the host's models when the server stops (default). Turn off when the <c>ModelHost</c> outlives the server and keeps
    /// being used (an embedding application that starts and stops the endpoints): draining stops the models' schedulers for good.
    /// </summary>
    public bool DrainModelsOnStop { get; set; } = true;

    /// <summary>
    /// Serve the host's task models (speech recognition on <c>/v1/audio/transcriptions</c>, speech synthesis on <c>/v1/audio/speech</c>,
    /// image generation on <c>/v1/images/generations</c>, time-series forecasting on <c>/v1/timeseries/forecast</c>) and its embedding and reranking models, looked up in the host per request by
    /// the <c>model</c> field. The endpoints are mapped whether or not such a model is registered yet, so models loaded after the
    /// server started are reachable at once (an application that loads and unloads models while serving). Default off: the endpoints
    /// exist only for models registered when they are mapped.
    /// </summary>
    public bool ServeHostModels { get; set; }
}
