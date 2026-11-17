using Microsoft.AspNetCore.Http;
using Sharplight.AspNetCore;

namespace Sharplight.Serving.Tests;

/// <summary>A fixed context per route group (the scope comes from the key).</summary>
public sealed class ScopeResolver(SharplightRequestContext template) : IModelEndpointResolver
{
    public ValueTask<SharplightRequestContext> ResolveAsync(HttpContext http, ApiKeyValidationResult authentication, CancellationToken cancellationToken)
        => ValueTask.FromResult(template with { Scope = authentication.Scope, Principal = authentication.Principal });
}
