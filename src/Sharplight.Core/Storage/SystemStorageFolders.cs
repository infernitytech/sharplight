namespace Sharplight.Storage;

/// <summary>The platform's per-user folders of <see cref="StorageMode.System"/>, each with a <c>sharplight</c> subfolder.</summary>
internal readonly record struct SystemStorageFolders(string Config, string Data, string Cache, string Logs)
{
    /// <summary>The folders of <paramref name="environment"/>'s platform.</summary>
    public static SystemStorageFolders For(StorageEnvironment environment)
    {
        var home = environment.HomeDirectory;
        switch (environment.Platform)
        {
            case StoragePlatform.Windows:
            {
                var roaming = environment.Variable("APPDATA") ?? Path.Combine(home, "AppData", "Roaming");
                var local = Path.Combine(environment.Variable("LOCALAPPDATA") ?? Path.Combine(home, "AppData", "Local"), StorageLayout.ProductDirectoryName);
                return new SystemStorageFolders(Path.Combine(roaming, StorageLayout.ProductDirectoryName), local, Path.Combine(local, StorageLayout.CacheDirectoryName),
                    Path.Combine(local, StorageLayout.LogsDirectoryName));
            }

            case StoragePlatform.MacOS:
            {
                var library = Path.Combine(home, "Library");
                var support = Path.Combine(library, "Application Support", StorageLayout.ProductDirectoryName);
                return new SystemStorageFolders(Path.Combine(support, StorageLayout.ConfigDirectoryName), support, Path.Combine(library, "Caches", StorageLayout.ProductDirectoryName),
                    Path.Combine(library, "Logs", StorageLayout.ProductDirectoryName));
            }

            default:
            {
                string Xdg(string variable, params string[] fallback) =>
                    Path.Combine(environment.Variable(variable) is { } value && Path.IsPathRooted(value) ? value : Path.Combine([home, .. fallback]),
                        StorageLayout.ProductDirectoryName);
                return new SystemStorageFolders(Xdg("XDG_CONFIG_HOME", ".config"), Xdg("XDG_DATA_HOME", ".local", "share"), Xdg("XDG_CACHE_HOME", ".cache"),
                    Path.Combine(Xdg("XDG_STATE_HOME", ".local", "state"), StorageLayout.LogsDirectoryName));
            }
        }
    }
}
