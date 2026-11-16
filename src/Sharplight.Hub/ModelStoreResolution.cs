using Sharplight.Storage;

namespace Sharplight.Hub;

/// <summary>The model store directory in effect and why (see <see cref="ModelStore.Resolve(string?)"/>).</summary>
/// <param name="Root">The store directory.</param>
/// <param name="Source">What decided it.</param>
/// <param name="Location">The kind of location: <see cref="ModelStoreLocation.Custom"/> for the command line and the environment.</param>
/// <param name="StorageMode">For <see cref="ModelStoreLocation.Application"/>: the mode of the storage layout the store belongs to.</param>
/// <param name="Notice">
/// Why the storage layout differs from what was asked for (a portable folder that is not writable), or null.
/// </param>
public sealed record ModelStoreResolution(string Root, ModelStoreSource Source, ModelStoreLocation Location, StorageMode StorageMode = StorageMode.System,
    string? Notice = null)
{
    /// <summary>A short description of where the store comes from, for settings pages and command output.</summary>
    public string Description
    {
        get
        {
            var origin = Source switch
            {
                ModelStoreSource.CommandLine => "--models-dir",
                ModelStoreSource.EnvironmentVariable => ModelStore.EnvironmentVariable,
                ModelStoreSource.Settings => "Settings",
                _ => "Default",
            };

            if (Source is ModelStoreSource.CommandLine or ModelStoreSource.EnvironmentVariable)
            {
                return origin;
            }

            var location = Location switch
            {
                ModelStoreLocation.Application => StorageMode switch
                {
                    StorageMode.Portable => "portable storage",
                    StorageMode.Custom => "custom storage folder",
                    _ => "system storage",
                },
                ModelStoreLocation.User => "system folder",
                _ => "custom folder",
            };
            return Notice is null ? $"{origin}: {location}" : $"{origin}: {location}; {Notice}";
        }
    }
}
