using System.Security.Cryptography;
using System.Text;
using Microsoft.AspNetCore.Http;

namespace Sharplight.AspNetCore;

/// <summary>
/// Identifies who a request is charged to for fair queuing (<see cref="Llm.Serving.GenerationRequest.Tenant"/>): the
/// client-supplied end-user id (OpenAI <c>user</c>, Anthropic <c>metadata.user_id</c>), else the API key, else nobody.
/// </summary>
internal static class Tenants
{
    /// <summary>The API key of the request (<c>Authorization: Bearer</c> or <c>x-api-key</c>), or null.</summary>
    public static string? PresentedKey(HttpContext http)
    {
        string auth = http.Request.Headers.Authorization.ToString();
        if (auth.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
        {
            return auth[7..].Trim();
        }

        return http.Request.Headers.TryGetValue("x-api-key", out var k) ? k.ToString() : null;
    }

    /// <summary>The tenant of a request; API keys are hashed so they never appear in memory dumps or logs.</summary>
    public static string? Of(HttpContext http, string? user)
    {
        if (!string.IsNullOrEmpty(user))
        {
            return "user:" + user;
        }

        return PresentedKey(http) is { Length: > 0 } key
            ? "key:" + Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(key)))[..16]
            : null;
    }
}
