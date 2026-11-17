using Microsoft.AspNetCore.Http;

namespace Sharplight.AspNetCore;

/// <summary>
/// Authenticates the API requests of the Sharplight endpoints. The default (<see cref="StaticApiKeyValidator"/>) checks
/// <see cref="SharplightServerOptions.ApiKeys"/>; register another implementation (or set one per route group with
/// <see cref="SharplightEndpointOptions.KeyValidator"/>) to check keys against a store. Hosts that authenticate with ASP.NET Core
/// themselves can read <c>HttpContext.User</c> here.
/// </summary>
public interface IApiKeyValidator
{
    /// <summary>Validates the presented key.</summary>
    /// <param name="presentedKey">The key of <c>Authorization: Bearer</c> or <c>x-api-key</c>, or null when none was sent.</param>
    /// <param name="http">The request.</param>
    /// <param name="cancellationToken">The request's cancellation.</param>
    /// <returns>The decision; <see cref="ApiKeyValidationResult.Invalid"/> answers 401.</returns>
    public ValueTask<ApiKeyValidationResult> ValidateAsync(string? presentedKey, HttpContext http, CancellationToken cancellationToken);
}
