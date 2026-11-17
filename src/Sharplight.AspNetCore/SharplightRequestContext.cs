namespace Sharplight.AspNetCore;

/// <summary>
/// Who a request is for and what it may use, as the <see cref="IModelEndpointResolver"/> decides it per request: the isolation scope,
/// the principal, the visible models (aliases), defaults, allowed tools and the fair-share tenant. Read it in handlers with
/// <c>HttpContext.GetSharplightRequestContext()</c>.
/// </summary>
public sealed record SharplightRequestContext
{
    /// <summary>
    /// Opaque isolation scope (a tenant, a space, an endpoint): partitions the engine's prefix cache
    /// (<c>GenerationRequest.CachePartition</c>) so identical prompts of different scopes never share cache entries, and tags usage
    /// and metrics. Null is the single default scope.
    /// </summary>
    public string? Scope { get; init; }

    /// <summary>Who sends the request (a user or key id from authentication), for usage records and stored state; null = anonymous.</summary>
    public string? Principal { get; init; }

    /// <summary>
    /// Who the request is charged to for fair queuing among requests of equal priority; null = derived from the scope and the
    /// client-supplied end-user id or the API key.
    /// </summary>
    public string? Tenant { get; init; }

    /// <summary>
    /// The models this request may use: the name clients send (alias) to the registered model name of the host. Null = every model the
    /// server exposes (<see cref="SharplightServerOptions.ModelFilter"/>). Names outside the map are answered with 404, as for unknown
    /// models; <c>/v1/models</c> lists only the aliases.
    /// </summary>
    public IReadOnlyDictionary<string, string>? Models { get; init; }

    /// <summary>The alias used when a request names no model (default: the first alias of the requested kind).</summary>
    public string? DefaultModel { get; init; }

    /// <summary>Defaults and limits of the request.</summary>
    public SharplightRequestDefaults? Defaults { get; init; }

    /// <summary>Names of the tools a request may declare; null = any, empty = none. Other tools are refused with 400.</summary>
    public IReadOnlySet<string>? AllowedTools { get; init; }

    /// <summary>Free data of the resolver (e.g. an endpoint or key id), passed on to usage observers and admission hooks.</summary>
    public IReadOnlyDictionary<string, object?> Properties { get; init; } = new Dictionary<string, object?>();
}
