using Sharplight.Hub.Catalog;

namespace Sharplight.Studio.Services.Library;

/// <summary>
/// The licence notices of the verified catalog: models whose entry has a non-permissive licence (a custom agreement, non-commercial use, use restrictions: OmniVoice, XTTS,
/// the CC-BY-NC models, Gemma, Llama, FLUX) show the entry's <c>licenseNotice</c>, or a generic line when it has none, as a callout once per model (until the user
/// dismisses it) on its Discover page and on the page of its kind. There is no acceptance step.
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
            return match is not null && For(match) is { } text ? new LicenseNotice(match.Id, text) : null;
        }
        catch (Exception error) when (error is IOException or InvalidDataException or InvalidOperationException or UnauthorizedAccessException)
        {
            return null;
        }
    }
}
