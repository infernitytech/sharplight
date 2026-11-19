using Sharplight.Hub.Catalog;

namespace Sharplight.Studio.Services.Library;

/// <summary>
/// The licence notices of the verified catalog: models whose entry has the license <c>other</c> or a non-commercial licence (OmniVoice, XTTS, CC-BY-NC models such as
/// Depth Anything 3 large, the FLUX non-commercial models) show the entry's notes prominently on their Discover page and while they are in use. There is no acceptance step.
/// </summary>
internal static class LicenseNotices
{
    /// <summary>Whether a licence needs a notice: <c>other</c> (a custom agreement) or a non-commercial one.</summary>
    /// <param name="license">The catalog entry's licence id, or null.</param>
    public static bool Requires(string? license)
    {
        if (string.IsNullOrWhiteSpace(license))
        {
            return false;
        }

        return license.Equals("other", StringComparison.OrdinalIgnoreCase) || license.Contains("non-commercial", StringComparison.OrdinalIgnoreCase)
            || license.Contains("noncommercial", StringComparison.OrdinalIgnoreCase) || license.Contains("-nc", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>The notice of an entry: its licence and its notes, or <see langword="null"/> when the licence needs none.</summary>
    /// <param name="entry">The catalog entry.</param>
    public static string? For(CatalogEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry);
        if (!Requires(entry.License))
        {
            return null;
        }

        string notes = string.IsNullOrWhiteSpace(entry.Notes) ? "This model has its own licence terms: read them on its page before you use it, especially for commercial use." : entry.Notes.Trim();
        return $"Licence: {entry.License}. {notes}";
    }

    /// <summary>
    /// The notice of the catalog entry a model on disk was installed from (the manifest's catalog key, or the repository of one of its variants), or
    /// <see langword="null"/> when there is none or the model did not come from the catalog.
    /// </summary>
    /// <param name="services">The services.</param>
    /// <param name="modelPath">The model's file or folder.</param>
    /// <param name="cancellationToken">Cancels the lookup.</param>
    public static async Task<string?> ForModelAsync(StudioServices services, string modelPath, CancellationToken cancellationToken = default)
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
            return match is null ? null : For(match);
        }
        catch (Exception error) when (error is IOException or InvalidDataException or InvalidOperationException or UnauthorizedAccessException)
        {
            return null;
        }
    }
}
