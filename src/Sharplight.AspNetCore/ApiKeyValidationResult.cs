namespace Sharplight.AspNetCore;

/// <summary>What an <see cref="IApiKeyValidator"/> decided about the presented key.</summary>
/// <param name="IsValid">The request may proceed.</param>
/// <param name="Principal">Who the key belongs to (a user or key id), or null.</param>
/// <param name="Scope">The isolation scope the key is bound to, or null (the resolver decides).</param>
public sealed record ApiKeyValidationResult(bool IsValid, string? Principal = null, string? Scope = null)
{
    /// <summary>No key is required: the request proceeds anonymously.</summary>
    public static ApiKeyValidationResult Anonymous { get; } = new(true);

    /// <summary>The key is missing or not accepted (HTTP 401).</summary>
    public static ApiKeyValidationResult Invalid { get; } = new(false);
}
