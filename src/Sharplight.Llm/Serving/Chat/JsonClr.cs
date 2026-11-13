using System.Text.Json;
using Sharplight.Templating;

namespace Sharplight.Llm.Serving.Chat;

/// <summary>JSON to template-value conversion (objects become <see cref="JinjaDict"/>, numbers long/double).</summary>
public static class JsonClr
{
    public static object? Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return Convert(doc.RootElement);
    }

    /// <summary>Parses a JSON object text into a dictionary; non-object or invalid text gives an empty dictionary.</summary>
    public static IReadOnlyDictionary<string, object?> ParseObject(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return new Dictionary<string, object?>();
        }

        try
        {
            if (Parse(json) is JinjaDict d)
            {
                return d.ToDictionary(kv => (string)kv.Key, kv => kv.Value);
            }
        }
        catch (JsonException) { }
        return new Dictionary<string, object?>();
    }

    public static object? Convert(JsonElement e)
    {
        switch (e.ValueKind)
        {
            case JsonValueKind.Object:
            {
                var d = new Dictionary<string, object?>();
                foreach (var p in e.EnumerateObject())
                    {
                        d[p.Name] = Convert(p.Value);
                    }

                    return JinjaValue.From(d);
            }
            case JsonValueKind.Array:
                return e.EnumerateArray().Select(Convert).ToList();
            case JsonValueKind.String:
                return e.GetString();
            case JsonValueKind.Number:
                return e.TryGetInt64(out var l) ? l : e.GetDouble();
            case JsonValueKind.True:
                return true;
            case JsonValueKind.False:
                return false;
            default:
                return null;
        }
    }
}
