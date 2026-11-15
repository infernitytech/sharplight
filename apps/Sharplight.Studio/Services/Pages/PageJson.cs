using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace Sharplight.Studio.Services.Pages;

/// <summary>Reads and writes the JSON files of the inference pages; a missing or damaged file reads as empty.</summary>
internal static class PageJson
{
    /// <summary>Reads <paramref name="path"/>, or returns a new instance when it is missing or not valid.</summary>
    public static T Read<T>(string path, JsonTypeInfo<T> info)
        where T : class, new()
    {
        try
        {
            return File.Exists(path) ? JsonSerializer.Deserialize(File.ReadAllText(path), info) ?? new T() : new T();
        }
        catch (Exception error) when (error is JsonException or IOException or UnauthorizedAccessException)
        {
            return new T();
        }
    }

    /// <summary>Writes <paramref name="value"/> to <paramref name="path"/> (through a temporary file, so a crash keeps the old content).</summary>
    public static void Write<T>(string path, T value, JsonTypeInfo<T> info)
    {
        string temporary = path + ".tmp";
        File.WriteAllText(temporary, JsonSerializer.Serialize(value, info));
        File.Move(temporary, path, overwrite: true);
    }
}
