using Sharplight.Hub;

namespace Sharplight.Studio.Services.Settings;

/// <summary>
/// Resolves the model store like the command-line tools: <c>--models-dir</c>, then the directory chosen in the settings, then
/// <c>SHARPLIGHT_MODELS</c>, then <c>models</c> next to the executable (portable), then the per-user store.
/// </summary>
internal static class StoreLocation
{
    /// <summary>The store root and a description of where it came from.</summary>
    public static (string Root, string Source) Resolve(string? commandLine, string? fromSettings, string? fromEnvironment, string executableDirectory, string userRoot)
    {
        if (!string.IsNullOrWhiteSpace(commandLine))
        {
            return (commandLine, "--models-dir");
        }

        if (!string.IsNullOrWhiteSpace(fromSettings))
        {
            return (fromSettings, "Settings");
        }

        if (!string.IsNullOrWhiteSpace(fromEnvironment))
        {
            return (fromEnvironment, ModelStore.EnvironmentVariable);
        }

        var portable = Path.Combine(executableDirectory, "models");
        return Directory.Exists(portable) ? (portable, "Portable (models next to the app)") : (userRoot, "Per-user default (shared with the CLI)");
    }

    /// <summary>Resolves with this process's environment.</summary>
    public static (string Root, string Source) Resolve(string? commandLine, string? fromSettings)
    {
        return Resolve(commandLine, fromSettings, Environment.GetEnvironmentVariable(ModelStore.EnvironmentVariable), AppContext.BaseDirectory, ModelStore.UserRoot);
    }
}
