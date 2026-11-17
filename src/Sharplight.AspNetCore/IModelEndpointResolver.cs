using Microsoft.AspNetCore.Http;

namespace Sharplight.AspNetCore;

/// <summary>
/// Decides per request which models, defaults and tools apply and in which isolation scope it runs (multi-tenant hosts: per route
/// group, endpoint, key or space). Called after authentication for every API request of the Sharplight endpoints. The default
/// (<see cref="DefaultModelEndpointResolver"/>) exposes the server's models with the global defaults in one scope. Register an
/// implementation in the container, or set one per route group with <see cref="SharplightEndpointOptions.Resolver"/>.
/// </summary>
public interface IModelEndpointResolver
{
    /// <summary>Resolves the request.</summary>
    /// <param name="http">The request.</param>
    /// <param name="authentication">The result of the <see cref="IApiKeyValidator"/>.</param>
    /// <param name="cancellationToken">The request's cancellation.</param>
    /// <returns>The request's context. Throw an exception to refuse it (it is answered with 500), or return a context with an empty
    /// <see cref="SharplightRequestContext.Models"/> to answer every model with 404.</returns>
    public ValueTask<SharplightRequestContext> ResolveAsync(HttpContext http, ApiKeyValidationResult authentication, CancellationToken cancellationToken);
}
