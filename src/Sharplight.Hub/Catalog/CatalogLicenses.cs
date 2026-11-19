namespace Sharplight.Hub.Catalog;

/// <summary>Which catalog licence ids are permissive, so need no licence notice.</summary>
public static class CatalogLicenses
{
    private static readonly HashSet<string> s_permissive = new(StringComparer.OrdinalIgnoreCase)
    {
        "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0", "cc-by-4.0", "cc-by-sa-4.0", "cdla-permissive-2.0", "bsl-1.0", "zlib",
    };

    /// <summary>Whether <paramref name="license"/> is a known permissive licence; false for unknown, custom (<c>other</c>), non-commercial and use-restricted ones.</summary>
    /// <param name="license">The catalog entry's licence id, or null.</param>
    public static bool IsPermissive(string? license) => !string.IsNullOrWhiteSpace(license) && s_permissive.Contains(license.Trim());

    /// <summary>Whether a licence is recorded and not permissive, so the entry needs a <see cref="CatalogEntry.LicenseNotice"/>.</summary>
    /// <param name="license">The catalog entry's licence id, or null.</param>
    public static bool NeedsNotice(string? license) => !string.IsNullOrWhiteSpace(license) && !IsPermissive(license);
}
