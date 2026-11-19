using System.Security.Cryptography;
using System.Text;
using Sharplight.Hub;
using Sharplight.Hub.Catalog;
using Sharplight.ModelFeed;

namespace Sharplight.Studio.Services.Library;

/// <summary>
/// The licence notices of the verified catalog: models whose entry has a non-permissive licence (a custom agreement, non-commercial use, use restrictions: OmniVoice, XTTS,
/// the CC-BY-NC models, Gemma, Llama, FLUX) show the entry's <c>licenseNotice</c>, or a generic line when it has none, as a callout once per model (until the user
/// dismisses it) on its Discover page and on the page of its kind. Before such a model is downloaded (catalog, feed or Hub) the user must accept its terms
/// (<see cref="ILicenseGate"/>); a model whose terms were accepted does not show the callout afterwards. <see cref="CatalogLicenses"/> is the one list of
/// permissive licences.
/// </summary>
internal static class LicenseNotices
{
    /// <summary>The line for a non-permissive licence whose entry has no <c>licenseNotice</c>.</summary>
    public const string GenericLine = "This model has its own licence terms: read them on its page before you use it, especially for commercial use.";

    /// <summary>Whether a licence needs a notice: any recorded licence that is not a known permissive one.</summary>
    /// <param name="license">The catalog entry's licence id, or null.</param>
    public static bool Requires(string? license) => CatalogLicenses.NeedsNotice(license);

    /// <summary>The notice of an entry: its <c>licenseNotice</c>, else a generic line naming the licence when that is not permissive, else <see langword="null"/>.</summary>
    /// <param name="entry">The catalog entry.</param>
    public static string? For(CatalogEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry);
        if (!string.IsNullOrWhiteSpace(entry.LicenseNotice))
        {
            return entry.LicenseNotice.Trim();
        }

        return Requires(entry.License) ? $"Licence: {entry.License}. {GenericLine}" : null;
    }

    /// <summary>The terms to accept before installing a variant of a catalog entry, or null for a permissive licence without notice.</summary>
    /// <param name="entry">The catalog entry.</param>
    /// <param name="variant">The variant being installed (its repository is linked), or null for the default one.</param>
    public static LicenseTerms? TermsFor(CatalogEntry entry, CatalogVariant? variant = null)
    {
        ArgumentNullException.ThrowIfNull(entry);
        if (For(entry) is not { } notice)
        {
            return null;
        }

        string? repository = (variant ?? entry.DefaultVariant)?.Repository;
        return new LicenseTerms(entry.Id, entry.Name, entry.License ?? "custom licence", notice, repository is null ? null : HubUrl(repository), Fingerprint(entry.License, notice));
    }

    /// <summary>The terms to accept before installing a package of the model feed, or null when its licence is permissive and not gated.</summary>
    /// <param name="packageId">The package id.</param>
    /// <param name="name">The package's name.</param>
    /// <param name="license">The package's licence, or null.</param>
    public static LicenseTerms? TermsFor(string packageId, string name, PackageLicense? license)
    {
        string? id = license?.Spdx ?? license?.Name;
        if (license is null || !(license.Gated || license.CommercialUse == false || Requires(id)))
        {
            return null;
        }

        string notice = license.Attribution is { Length: > 0 } attribution ? attribution
            : license.CommercialUse == false ? $"Licence: {license.Name ?? id}. Commercial use is not allowed. {GenericLine}"
            : $"Licence: {license.Name ?? id}. {GenericLine}";
        return new LicenseTerms("feed:" + packageId, name, license.Name ?? id ?? "custom licence", notice, license.Url, Fingerprint(id, notice, license.Url));
    }

    /// <summary>The terms to accept before downloading a Hub repository, or null when its licence is permissive (or not recorded) and it is not gated.</summary>
    /// <param name="repository">The Hub repository (<c>owner/name</c>).</param>
    /// <param name="license">The Hub's licence tag, or null.</param>
    /// <param name="gated">The repository is gated.</param>
    public static LicenseTerms? TermsFor(string repository, string? license, bool gated)
    {
        if (!gated && !Requires(license))
        {
            return null;
        }

        string name = license ?? "gated";
        string notice = $"Licence: {name}. {(gated ? "The repository is gated: its terms are accepted on the Hub. " : "")}{GenericLine}";
        return new LicenseTerms("hub:" + repository, repository, name, notice, HubUrl(repository), Fingerprint(license, notice));
    }

    /// <summary>Whether the user dismissed the notice, or accepted its terms before the download.</summary>
    /// <param name="services">The services.</param>
    /// <param name="notice">The notice.</param>
    public static bool IsHandled(StudioServices services, LicenseNotice notice)
    {
        ArgumentNullException.ThrowIfNull(notice);
        return IsDismissed(services, notice.EntryId) || services.Licenses.HasAccepted(notice.EntryId)
            || (notice.AcceptedKeys?.Any(services.Licenses.HasAccepted) ?? false);
    }

    private static string HubUrl(string repository) => new Uri(HubClientOptions.DefaultEndpoint, repository).ToString();

    private static string Fingerprint(params string?[] parts)
        => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(string.Join('\n', parts))))[..12].ToLowerInvariant();

    /// <summary>Whether the user dismissed the notice of the catalog entry <paramref name="entryId"/>.</summary>
    /// <param name="services">The services.</param>
    /// <param name="entryId">The catalog entry's id.</param>
    public static bool IsDismissed(StudioServices services, string entryId)
    {
        ArgumentNullException.ThrowIfNull(services);
        return services.Settings.Current.DismissedLicenseNotices.Contains(entryId, StringComparer.Ordinal);
    }

    /// <summary>Remembers in the settings that the notice of <paramref name="entryId"/> was dismissed.</summary>
    /// <param name="services">The services.</param>
    /// <param name="entryId">The catalog entry's id.</param>
    public static void Dismiss(StudioServices services, string entryId)
    {
        ArgumentNullException.ThrowIfNull(services);
        services.Settings.Update(settings => settings.DismissedLicenseNotices.Contains(entryId, StringComparer.Ordinal)
            ? settings
            : settings with { DismissedLicenseNotices = [.. settings.DismissedLicenseNotices, entryId] });
    }

    /// <summary>
    /// The notice (and the key of its dismissal) of the catalog entry a model on disk was installed from (the manifest's catalog key, or the repository of one of its variants), or
    /// <see langword="null"/> when there is none or the model did not come from the catalog. Its <see cref="LicenseNotice.EntryId"/> is the key of the dismissal.
    /// </summary>
    /// <param name="services">The services.</param>
    /// <param name="modelPath">The model's file or folder.</param>
    /// <param name="cancellationToken">Cancels the lookup.</param>
    public static async Task<LicenseNotice?> FindAsync(StudioServices services, string modelPath, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(services);
        try
        {
            var library = await services.Library.ListAsync(cancellationToken).ConfigureAwait(false);
            if (library.FirstOrDefault(entry => string.Equals(entry.Path, modelPath, StringComparison.Ordinal))?.Manifest is not { } manifest)
            {
                return null;
            }

            var catalog = await services.Hub.GetCatalogAsync(cancellationToken).ConfigureAwait(false);
            CatalogEntry? match = null;
            if (manifest.Catalog is { Length: > 0 } key)
            {
                string id = key.Contains('/', StringComparison.Ordinal) ? key[..key.IndexOf('/', StringComparison.Ordinal)] : key;
                match = catalog.Entries.FirstOrDefault(entry => entry.Id == id);
            }

            match ??= catalog.Entries.FirstOrDefault(entry => entry.Variants.Any(variant => string.Equals(variant.Repository, manifest.Repository, StringComparison.OrdinalIgnoreCase)));
            List<string> keys = [];
            if (manifest.FeedPackage is { } package)
            {
                keys.Add("feed:" + package);
            }

            keys.Add("hub:" + manifest.Repository);
            return match is not null && For(match) is { } text ? new LicenseNotice(match.Id, text, keys) : null;
        }
        catch (Exception error) when (error is IOException or InvalidDataException or InvalidOperationException or UnauthorizedAccessException)
        {
            return null;
        }
    }
}
