using System.Reflection;

namespace Sharplight.Studio.Services.Updates;

/// <summary>
/// Where the update feed is. Order of precedence: the <c>SHARPLIGHT_STUDIO_UPDATE_URL</c> environment variable, the
/// <c>UpdateUrl</c> setting, then the URL baked into the build (MSBuild property <c>StudioUpdateUrl</c>, assembly metadata
/// <c>StudioUpdateUrl</c>; CI takes it from the repository variable <c>STUDIO_UPDATE_URL</c>).
/// </summary>
internal static class UpdateEndpoint
{
    /// <summary>The environment variable that overrides everything else.</summary>
    public const string EnvironmentVariable = "SHARPLIGHT_STUDIO_UPDATE_URL";

    /// <summary>The key of the assembly metadata that holds the URL baked into the build.</summary>
    public const string MetadataKey = "StudioUpdateUrl";

    /// <summary>The URL baked into <paramref name="assembly"/>, or null.</summary>
    public static string? Baked(Assembly assembly) => assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
        .FirstOrDefault(attribute => attribute.Key == MetadataKey)?.Value;

    /// <summary>Picks the first usable (absolute http or https) URL of the three sources; null when none is.</summary>
    public static string? Resolve(string? environment, string? setting, string? baked)
    {
        foreach (var candidate in new[] { environment, setting, baked })
        {
            var url = candidate?.Trim();
            if (!string.IsNullOrEmpty(url) && Uri.TryCreate(url, UriKind.Absolute, out var uri)
                && (uri.Scheme == Uri.UriSchemeHttps || uri.Scheme == Uri.UriSchemeHttp))
            {
                return url.TrimEnd('/') + "/";
            }
        }

        return null;
    }
}
