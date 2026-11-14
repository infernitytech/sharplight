using Sharplight.Hub;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Settings;

/// <summary>
/// Resolves the model store like the command-line tools: <c>--models-dir</c>, then <c>SHARPLIGHT_MODELS</c>, then the location
/// chosen in the settings, then the <c>data</c> folder in the application directory (the per-user store when that is not writable).
/// </summary>
internal static class StoreLocation
{
    /// <summary>The store choice of <paramref name="settings"/>, or null when none was made (the default applies).</summary>
    public static ModelStoreSettings? ToStoreSettings(AppSettings settings)
    {
        var custom = string.IsNullOrWhiteSpace(settings.ModelsDirectory) ? null : settings.ModelsDirectory.Trim();
        var location = settings.ModelsLocation ?? (custom is null ? null : ModelStoreLocation.Custom);
        return location is null ? null : new ModelStoreSettings { Location = location.Value, CustomPath = custom };
    }

    /// <summary>The location to show in the settings: the chosen one, or the application folder when none was chosen.</summary>
    public static ModelStoreLocation Effective(AppSettings settings) => ToStoreSettings(settings)?.Location ?? ModelStoreLocation.Application;

    /// <summary>Resolves with this process's environment and application directory.</summary>
    public static ModelStoreResolution Resolve(string? commandLine, AppSettings settings)
    {
        return ModelStore.Resolve(commandLine, ToStoreSettings(settings));
    }
}
