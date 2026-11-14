namespace Sharplight.Hub;

/// <summary>The model store directory in effect and why (see <see cref="ModelStore.Resolve(string?)"/>).</summary>
/// <param name="Root">The store directory.</param>
/// <param name="Source">What decided it.</param>
/// <param name="Location">The kind of location: <see cref="ModelStoreLocation.Custom"/> for the command line and the environment.</param>
/// <param name="ApplicationFolderNotWritable">
/// The application folder was asked for but is not writable (a system-wide install, a read-only mount or an app bundle), so
/// <see cref="Root"/> is the per-user store.
/// </param>
public sealed record ModelStoreResolution(string Root, ModelStoreSource Source, ModelStoreLocation Location, bool ApplicationFolderNotWritable = false)
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

            if (ApplicationFolderNotWritable)
            {
                return $"{origin}: user folder, the application folder is not writable";
            }

            if (Source is ModelStoreSource.CommandLine or ModelStoreSource.EnvironmentVariable)
            {
                return origin;
            }

            var location = Location switch
            {
                ModelStoreLocation.Application => "application folder",
                ModelStoreLocation.User => "user folder",
                _ => "custom folder",
            };
            return $"{origin}: {location}";
        }
    }
}
