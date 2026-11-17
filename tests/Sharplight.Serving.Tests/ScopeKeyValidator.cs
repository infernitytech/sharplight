using Microsoft.AspNetCore.Http;
using Sharplight.AspNetCore;

namespace Sharplight.Serving.Tests;

/// <summary>Accepts the keys of one scope only.</summary>
public sealed class ScopeKeyValidator(string scope, params string[] keys) : IApiKeyValidator
{
    public ValueTask<ApiKeyValidationResult> ValidateAsync(string? presentedKey, HttpContext http, CancellationToken cancellationToken)
        => ValueTask.FromResult(presentedKey is not null && keys.Contains(presentedKey)
            ? new ApiKeyValidationResult(true, "user:" + presentedKey, scope)
            : ApiKeyValidationResult.Invalid);
}
