using System.Security.Cryptography;
using System.Text;
using Microsoft.AspNetCore.Http;

namespace Sharplight.AspNetCore;

/// <summary>
/// Accepts a fixed list of keys (compared in constant time); an empty list lets every request in anonymously. The principal of an
/// accepted key is <c>key:</c> plus the first 16 hex digits of its SHA-256 (keys never appear in usage records or logs).
/// </summary>
/// <param name="keys">The accepted keys.</param>
public sealed class StaticApiKeyValidator(IEnumerable<string> keys) : IApiKeyValidator
{
    private readonly byte[][] _keys = [.. keys.Select(key => Encoding.UTF8.GetBytes(key))];

    /// <inheritdoc/>
    public ValueTask<ApiKeyValidationResult> ValidateAsync(string? presentedKey, HttpContext http, CancellationToken cancellationToken)
    {
        if (_keys.Length == 0)
        {
            return ValueTask.FromResult(ApiKeyValidationResult.Anonymous);
        }

        if (presentedKey is null)
        {
            return ValueTask.FromResult(ApiKeyValidationResult.Invalid);
        }

        var presented = Encoding.UTF8.GetBytes(presentedKey);
        bool ok = false;
        foreach (var key in _keys)
        {
            // Every key is compared (no early exit) so timing does not reveal which one matched.
            ok |= CryptographicOperations.FixedTimeEquals(presented, key);
        }

        return ValueTask.FromResult(ok ? new ApiKeyValidationResult(true, KeyPrincipal(presentedKey)) : ApiKeyValidationResult.Invalid);
    }

    /// <summary>The principal id of a key: <c>key:</c> and a prefix of its SHA-256.</summary>
    /// <param name="key">The key.</param>
    /// <returns>The id.</returns>
    public static string KeyPrincipal(string key) => "key:" + Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(key)))[..16];
}
