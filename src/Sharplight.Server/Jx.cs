using System.Text.Json;

namespace Sharplight.Server;

/// <summary>Typed accessors over a request JSON object.</summary>
internal static class Jx
{
    public static JsonElement? Get(this JsonElement o, string name)
        => o.ValueKind == JsonValueKind.Object && o.TryGetProperty(name, out var v) && v.ValueKind != JsonValueKind.Null ? v : null;
    public static string? Str(this JsonElement o, string name) => o.Get(name) is { ValueKind: JsonValueKind.String }
    v ? v.GetString() : null;
    public static int? Int(this JsonElement o, string name, string? param = null)
    {
        if (o.Get(name) is not { } v)
        {
            return null;
        }

        if (v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out var i))
        {
            return i;
        }

        if (v.ValueKind == JsonValueKind.Number && v.TryGetDouble(out var d))
        {
            return (int)d;
        }

        throw ApiException.BadRequest($"'{name}' must be an integer.", param ?? name);
    }
    public static float? Float(this JsonElement o, string name)
        => o.Get(name) is { }
        v ? v.ValueKind == JsonValueKind.Number ? v.GetSingle() : throw ApiException.BadRequest($"'{name}' must be a number.", name) : null;
    public static bool? Bool(this JsonElement o, string name)
        => o.Get(name) is { }
        v ? v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : throw ApiException.BadRequest($"'{name}' must be a boolean.", name) : null;
    public static IEnumerable<JsonElement> Arr(this JsonElement o, string name)
        => o.Get(name) is { }
        v ? v.ValueKind == JsonValueKind.Array ? v.EnumerateArray() : throw ApiException.BadRequest($"'{name}' must be an array.", name) : [];
    public static JsonElement? Obj(this JsonElement o, string name)
        => o.Get(name) is { ValueKind: JsonValueKind.Object }
        v ? v : null;
}
