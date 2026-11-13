using System.Text.Json;
using Sharplight.Samples;

namespace Sharplight.Serve;

/// <summary>Reads the JSON configuration file (comments and trailing commas are allowed).</summary>
internal static class ServerConfigLoader
{
    /// <exception cref="SampleException">The file is missing, empty or not valid JSON.</exception>
    public static ServerConfig Load(string path)
    {
        if (!File.Exists(path))
        {
            throw new SampleException($"Config file '{path}' does not exist.");
        }

        try
        {
            return JsonSerializer.Deserialize(File.ReadAllText(path), ServeJsonContext.Default.ServerConfig)
                ?? throw new SampleException($"Config file '{path}' is empty.");
        }
        catch (JsonException error)
        {
            throw new SampleException($"Config file '{path}' is not valid: {error.Message}", error);
        }
    }
}
