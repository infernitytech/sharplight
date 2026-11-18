using System.Reflection;
using System.Text.Json;

namespace Sharplight.Studio.Services.FirstRun;

/// <summary>Reads the starter preferences: the file in the settings folder when it is valid, else the resource built into Studio.</summary>
internal static class StarterPreferencesLoader
{
    /// <summary>The name of the embedded default.</summary>
    public const string ResourceName = "Sharplight.Studio.starter-models.json";

    /// <summary>The file name of the override in the settings folder.</summary>
    public const string FileName = "starter-models.json";

    /// <summary>Reads the preferences; a custom file that gives no example prompts uses the built-in ones.</summary>
    /// <param name="overrideFile">The override file, or null; an unreadable or invalid file is ignored.</param>
    public static StarterPreferences Load(string? overrideFile)
    {
        var builtIn = ReadEmbedded();
        if (overrideFile is not null && File.Exists(overrideFile))
        {
            try
            {
                var custom = Parse(File.ReadAllText(overrideFile));
                if (custom is { Roles.Count: > 0 })
                {
                    return custom.ExamplePrompts.Count > 0 ? custom : custom with { ExamplePrompts = builtIn.ExamplePrompts };
                }
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException or JsonException)
            {
                // The built-in preferences are used.
            }
        }

        return builtIn;
    }

    private static StarterPreferences ReadEmbedded()
    {
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(ResourceName)
            ?? throw new InvalidOperationException($"The embedded resource {ResourceName} is missing.");
        using var reader = new StreamReader(stream);
        return Parse(reader.ReadToEnd()) ?? throw new InvalidOperationException($"{ResourceName} is empty.");
    }

    internal static StarterPreferences? Parse(string json) => JsonSerializer.Deserialize(json, StarterJsonContext.Default.StarterPreferences);
}
