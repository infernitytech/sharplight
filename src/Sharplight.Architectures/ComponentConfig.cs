using System.Globalization;
using System.Text.Json;

namespace Sharplight.Architectures;

/// <summary>Builds the JSON configuration values of <see cref="Sharplight.Models.ModelComponent.Config"/> from plain values.</summary>
internal static class ComponentConfig
{
    /// <summary>The configuration for the given entries (numbers, booleans and strings).</summary>
    /// <param name="entries">Key and value pairs.</param>
    public static IReadOnlyDictionary<string, JsonElement> Of(params (string Key, object Value)[] entries)
    {
        var config = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
        foreach (var (key, value) in entries)
        {
            using var stream = new MemoryStream();
            using (var writer = new Utf8JsonWriter(stream))
            {
                switch (value)
                {
                    case string text:
                        writer.WriteStringValue(text);
                        break;
                    case bool flag:
                        writer.WriteBooleanValue(flag);
                        break;
                    case float number:
                        writer.WriteNumberValue(number);
                        break;
                    case double number:
                        writer.WriteNumberValue(number);
                        break;
                    case IConvertible number:
                        writer.WriteNumberValue(number.ToInt64(CultureInfo.InvariantCulture));
                        break;
                    default:
                        throw new ArgumentException($"Unsupported configuration value type {value.GetType()} for '{key}'.");
                }
            }

            using var document = JsonDocument.Parse(stream.ToArray());
            config[key] = document.RootElement.Clone();
        }

        return config;
    }
}
