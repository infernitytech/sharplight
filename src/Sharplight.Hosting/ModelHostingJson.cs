using System.Text.Json;

namespace Sharplight.Hosting;

/// <summary>
/// Reads and writes <see cref="ModelHostingOptions"/> as JSON (camelCase names, comments and trailing commas allowed; sizes as bytes or
/// <c>"2G"</c>). See docs/hosting.md for the format.
/// </summary>
public static class ModelHostingJson
{
    /// <summary>Parses the JSON form of the options.</summary>
    /// <param name="json">The JSON text.</param>
    /// <returns>The options.</returns>
    /// <exception cref="FormatException">The text is not valid hosting options.</exception>
    public static ModelHostingOptions Parse(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        try
        {
            return JsonSerializer.Deserialize(json, ModelHostingJsonContext.Default.ModelHostingOptions)
                ?? throw new FormatException("The hosting options are empty (null).");
        }
        catch (JsonException error)
        {
            throw new FormatException($"The hosting options are not valid: {error.Message}", error);
        }
    }

    /// <summary>Reads the options from a JSON file.</summary>
    /// <param name="path">The file.</param>
    /// <returns>The options.</returns>
    /// <exception cref="FileNotFoundException">The file does not exist.</exception>
    /// <exception cref="FormatException">The file is not valid hosting options.</exception>
    public static ModelHostingOptions Load(string path)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Hosting configuration '{path}' does not exist.", path);
        }

        try
        {
            return Parse(File.ReadAllText(path));
        }
        catch (FormatException error)
        {
            throw new FormatException($"Hosting configuration '{path}': {error.Message}", error);
        }
    }

    /// <summary>Writes the options as indented JSON (the delegates are not part of the JSON form).</summary>
    /// <param name="options">The options.</param>
    /// <returns>The JSON text.</returns>
    public static string Serialize(ModelHostingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return JsonSerializer.Serialize(options, ModelHostingJsonContext.Default.ModelHostingOptions);
    }

    /// <summary>Copies every JSON-backed setting of <paramref name="source"/> onto <paramref name="target"/> (the delegates are kept).</summary>
    /// <param name="source">The options read from JSON.</param>
    /// <param name="target">The options to change.</param>
    public static void CopyTo(ModelHostingOptions source, ModelHostingOptions target)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(target);
        target.Reserve = source.Reserve;
        target.Budget = source.Budget;
        target.MaxLoadedModels = source.MaxLoadedModels;
        target.IdleTimeoutSeconds = source.IdleTimeoutSeconds;
        target.Lazy = source.Lazy;
        target.LoadStartupModels = source.LoadStartupModels;
        target.OnDemand = source.OnDemand;
        target.DisabledFeatures = [.. source.DisabledFeatures];
        target.Defaults = source.Defaults;
        target.Models = [.. source.Models];
    }
}
