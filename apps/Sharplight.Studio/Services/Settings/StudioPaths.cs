namespace Sharplight.Studio.Services.Settings;

/// <summary>Where the app keeps its settings and data (<c>SHARPLIGHT_STUDIO_HOME</c> overrides both).</summary>
internal sealed record StudioPaths(string ConfigDirectory, string DataDirectory)
{
    /// <summary>Environment variable that moves settings and data into one directory (portable use, tests).</summary>
    public const string HomeVariable = "SHARPLIGHT_STUDIO_HOME";

    /// <summary>The settings file.</summary>
    public string SettingsFile => Path.Combine(ConfigDirectory, "settings.json");

    /// <summary>The conversation and statistics database.</summary>
    public string DatabaseFile => Path.Combine(DataDirectory, "studio.db");

    /// <summary>The per-user locations: <c>sharplight/studio</c> under the roaming and local application data directories.</summary>
    public static StudioPaths Default()
    {
        var home = Environment.GetEnvironmentVariable(HomeVariable);
        if (!string.IsNullOrWhiteSpace(home))
        {
            return new StudioPaths(home, home);
        }

        string Folder(Environment.SpecialFolder folder) =>
            Path.Combine(Environment.GetFolderPath(folder, Environment.SpecialFolderOption.DoNotVerify), "sharplight", "studio");
        return new StudioPaths(Folder(Environment.SpecialFolder.ApplicationData), Folder(Environment.SpecialFolder.LocalApplicationData));
    }
}
