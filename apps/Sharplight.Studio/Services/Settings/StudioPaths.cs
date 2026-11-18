using Sharplight.Storage;

namespace Sharplight.Studio.Services.Settings;

/// <summary>
/// Where the app keeps its own files: <c>studio</c> in the configuration and data folders of the storage layout
/// (<see cref="StorageLayout"/>), or one directory given by <c>SHARPLIGHT_STUDIO_HOME</c>.
/// </summary>
internal sealed record StudioPaths(string ConfigDirectory, string DataDirectory)
{
    /// <summary>Environment variable that moves settings and data into one directory (tests, side-by-side profiles).</summary>
    public const string HomeVariable = "SHARPLIGHT_STUDIO_HOME";

    /// <summary>Name of the app's folder in the storage layout's configuration and data folders.</summary>
    public const string ApplicationDirectoryName = "studio";

    /// <summary>The settings file.</summary>
    public string SettingsFile => Path.Combine(ConfigDirectory, "settings.json");

    /// <summary>The user's starter model preferences, which replace the built-in ones when present (see <see cref="FirstRun.StarterPreferencesLoader"/>).</summary>
    public string StarterModelsFile => Path.Combine(ConfigDirectory, "starter-models.json");

    /// <summary>The Hub's highlighted models, cached for Discover.</summary>
    public string HighlightsFile => Path.Combine(DataDirectory, "discover-highlights.json");

    /// <summary>The verified-model catalog last downloaded (see <see cref="Hub.VerifiedCatalogSource"/>).</summary>
    public string CatalogFile => Path.Combine(DataDirectory, "verified-models.json");

    /// <summary>The accepted sequence of every model feed (rollback protection).</summary>
    public string FeedStateFile => Path.Combine(DataDirectory, "feed-state.json");

    /// <summary>The last verified documents of every model feed channel (offline listing).</summary>
    public string FeedCacheDirectory => Path.Combine(DataDirectory, "feed-cache");

    /// <summary>The conversation and statistics database.</summary>
    public string DatabaseFile => Path.Combine(DataDirectory, "studio.db");

    /// <summary>The pages' presets, history and outputs.</summary>
    public string PagesDirectory => Path.Combine(DataDirectory, "pages");

    /// <summary>The generated images (and later other media), each in its dated folder with a JSON sidecar.</summary>
    public string GenerationsDirectory => Path.Combine(DataDirectory, "generations");

    /// <summary>The app's folders in <paramref name="layout"/>, or <paramref name="home"/> for both when it is set.</summary>
    /// <param name="layout">The storage layout.</param>
    /// <param name="home">The value of <see cref="HomeVariable"/>, or null.</param>
    public static StudioPaths From(StorageLayout layout, string? home)
    {
        if (!string.IsNullOrWhiteSpace(home))
        {
            return new StudioPaths(home, home);
        }

        return new StudioPaths(layout.GetConfigDirectory(ApplicationDirectoryName), layout.GetDataDirectory(ApplicationDirectoryName));
    }
}
