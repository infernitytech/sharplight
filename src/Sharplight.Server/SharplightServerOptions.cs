using Sharplight.Llm.Serving;

namespace Sharplight.Server;

/// <summary>
/// Settings of the HTTP server (see <see cref="SharplightServerExtensions"/>): which APIs are exposed, authentication, CORS,
/// request limits and default sampling. Configure through the callback of <c>AddSharplight</c>.
/// </summary>
public sealed class SharplightServerOptions
{
    /// <summary>Expose the OpenAI-compatible API (<c>/v1/chat/completions</c>, <c>/v1/completions</c>, <c>/v1/models</c>).</summary>
    public bool OpenAi { get; set; } = true;
    /// <summary>Expose the Anthropic-compatible API (<c>/v1/messages</c>, <c>/v1/messages/count_tokens</c>).</summary>
    public bool Anthropic { get; set; } = true;
    /// <summary>Accepted API keys (<c>Authorization: Bearer</c> or <c>x-api-key</c>). Empty = no authentication.</summary>
    public List<string> ApiKeys { get; set; } = [];
    /// <summary>Allowed CORS origins ("*" for any). Empty = CORS headers are not sent.</summary>
    public List<string> CorsOrigins { get; set; } = [];
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
}
