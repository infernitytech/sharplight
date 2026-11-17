namespace Sharplight.AspNetCore;

/// <summary>
/// Options of one <c>MapSharplightServer</c> call, so a host can map the APIs several times into different route groups (endpoints,
/// tenants) with their own authentication, resolver, admission and usage hooks. Unset hooks fall back to the services registered in
/// the container, then to the defaults built from <see cref="SharplightServerOptions"/>.
/// </summary>
public sealed class SharplightEndpointOptions
{
    /// <summary>Map the OpenAI-compatible API; null = <see cref="SharplightServerOptions.OpenAi"/>.</summary>
    public bool? OpenAi { get; set; }

    /// <summary>Map the Anthropic-compatible API; null = <see cref="SharplightServerOptions.Anthropic"/>.</summary>
    public bool? Anthropic { get; set; }

    /// <summary>Map the embedding and rerank endpoints (when embedding or rerank models are registered).</summary>
    public bool Retrieval { get; set; } = true;

    /// <summary>Map the audio and image endpoints (when such models are registered).</summary>
    public bool Media { get; set; } = true;

    /// <summary>Map <c>/health</c>, <c>/health/live</c>, <c>/health/ready</c> and <c>/metrics</c> (turn off for tenant route groups).</summary>
    public bool Operations { get; set; } = true;

    /// <summary>Map the <c>/admin</c> endpoints (when <see cref="SharplightServerOptions.Admin"/> is on; turn off for tenant route groups).</summary>
    public bool Admin { get; set; } = true;

    /// <summary>Authentication of these endpoints; null = the registered <see cref="IApiKeyValidator"/>, else the server's API keys.</summary>
    public IApiKeyValidator? KeyValidator { get; set; }

    /// <summary>The per-request resolver of these endpoints; null = the registered one, else <see cref="DefaultModelEndpointResolver"/>.</summary>
    public IModelEndpointResolver? Resolver { get; set; }

    /// <summary>The admission hook of these endpoints; null = the registered one, else none.</summary>
    public IRequestAdmission? Admission { get; set; }

    /// <summary>A usage observer of these endpoints, called in addition to the registered ones.</summary>
    public ISharplightUsageObserver? UsageObserver { get; set; }
}
