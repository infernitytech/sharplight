using Microsoft.AspNetCore.Http;

namespace Sharplight.AspNetCore;

/// <summary>
/// The single-host default: every model the server exposes, the global defaults of <see cref="SharplightServerOptions"/>, any tools;
/// the scope is the one the key is bound to (none for static keys) and the principal the key's.
/// </summary>
public sealed class DefaultModelEndpointResolver : IModelEndpointResolver
{
    /// <inheritdoc/>
    public ValueTask<SharplightRequestContext> ResolveAsync(HttpContext http, ApiKeyValidationResult authentication, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(authentication);
        return ValueTask.FromResult(new SharplightRequestContext { Scope = authentication.Scope, Principal = authentication.Principal });
    }
}
