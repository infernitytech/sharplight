using System.Text.RegularExpressions;

namespace Sharplight.ModelFeed;

/// <summary>Relative paths of the static feed layout (docs/model-feed.md) and package id rules.</summary>
public static partial class FeedLayout
{
    /// <summary>The service index.</summary>
    public const string ServiceIndex = "index.json";

    /// <summary>Signature of the service index.</summary>
    public const string ServiceIndexSignature = "index.sig";

    /// <summary>The key set.</summary>
    public const string KeySet = "keys/keyset.json";

    /// <summary>Root signature of the key set.</summary>
    public const string KeySetSignature = "keys/keyset.sig";

    /// <summary>The catalog index.</summary>
    public const string CatalogIndex = "catalog/index.json";

    /// <summary>The package index template.</summary>
    public const string PackageIndexTemplate = "packages/{id}/index.json";

    /// <summary>Name of a version's manifest.</summary>
    public const string ManifestFileName = "manifest.json";

    /// <summary>Name of a version's manifest signature.</summary>
    public const string ManifestSignatureFileName = "manifest.sig";

    /// <summary>Path of catalog page <paramref name="number"/> (1-based).</summary>
    /// <param name="number">Page number.</param>
    /// <returns>The relative path.</returns>
    public static string CatalogPage(int number) => $"catalog/page/{number}.json";

    /// <summary>Path of a package's index.</summary>
    /// <param name="id">Package id.</param>
    /// <returns>The relative path.</returns>
    public static string PackageIndex(string id) => $"packages/{RequireId(id)}/index.json";

    /// <summary>Folder of a package version.</summary>
    /// <param name="id">Package id.</param>
    /// <param name="version">Version.</param>
    /// <returns>The relative path.</returns>
    public static string VersionFolder(string id, string version) => $"packages/{RequireId(id)}/{RequireVersion(version)}";

    /// <summary>Path of a version's manifest.</summary>
    /// <param name="id">Package id.</param>
    /// <param name="version">Version.</param>
    /// <returns>The relative path.</returns>
    public static string Manifest(string id, string version) => $"{VersionFolder(id, version)}/{ManifestFileName}";

    /// <summary>Path of a version's manifest signature.</summary>
    /// <param name="id">Package id.</param>
    /// <param name="version">Version.</param>
    /// <returns>The relative path.</returns>
    public static string ManifestSignature(string id, string version) => $"{VersionFolder(id, version)}/{ManifestSignatureFileName}";

    /// <summary>Path of a file of a version.</summary>
    /// <param name="id">Package id.</param>
    /// <param name="version">Version.</param>
    /// <param name="fileName">File name from the manifest.</param>
    /// <returns>The relative path.</returns>
    public static string VersionFile(string id, string version, string fileName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(fileName);
        if (fileName.Contains('/', StringComparison.Ordinal) || fileName.Contains('\\', StringComparison.Ordinal) || fileName.StartsWith('.'))
        {
            throw new FeedFormatException($"'{fileName}' is not a plain file name.");
        }

        return $"{VersionFolder(id, version)}/{fileName}";
    }

    /// <summary>The conventional <c>.slm</c> file name of a version.</summary>
    /// <param name="id">Package id.</param>
    /// <param name="version">Version.</param>
    /// <returns>The file name.</returns>
    public static string PackageFileName(string id, string version) => $"{RequireId(id)}.{RequireVersion(version)}.slm";

    /// <summary>Whether <paramref name="id"/> is a valid package id (lower case letters, digits, dots and hyphens; starts with a letter or digit).</summary>
    /// <param name="id">The id.</param>
    /// <returns><see langword="true"/> when valid.</returns>
    public static bool IsValidId(string? id) => id is not null && IdPattern().IsMatch(id);

    private static string RequireId(string id)
    {
        return IsValidId(id) ? id : throw new FeedFormatException($"'{id}' is not a valid package id.");
    }

    private static string RequireVersion(string version)
    {
        return Versioning.SemanticVersion.TryParse(version, out _) && !version.Contains('/', StringComparison.Ordinal)
            ? version
            : throw new FeedFormatException($"'{version}' is not a semantic version.");
    }

    [GeneratedRegex("^[a-z0-9][a-z0-9.-]{0,199}$")]
    private static partial Regex IdPattern();
}
