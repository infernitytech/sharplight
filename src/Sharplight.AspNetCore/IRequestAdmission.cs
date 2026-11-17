using Microsoft.AspNetCore.Http;

namespace Sharplight.AspNetCore;

/// <summary>
/// Quota, rate-limit and concurrency hook: called once the model of a request is known, before it runs. Throw an
/// <see cref="ApiException"/> (e.g. <see cref="ApiException.TooManyRequests"/>) to refuse the request; return a handle to hold a
/// concurrency slot, which is disposed when the response is complete. Register one in the container or per route group
/// (<see cref="SharplightEndpointOptions.Admission"/>).
/// </summary>
public interface IRequestAdmission
{
    /// <summary>Admits or refuses a request.</summary>
    /// <param name="context">The request's context (scope, principal, properties).</param>
    /// <param name="model">The model name the client used (the alias).</param>
    /// <param name="http">The request.</param>
    /// <param name="cancellationToken">The request's cancellation.</param>
    /// <returns>A handle released after the response, or null.</returns>
    public ValueTask<IAsyncDisposable?> AdmitAsync(SharplightRequestContext context, string model, HttpContext http, CancellationToken cancellationToken);
}
