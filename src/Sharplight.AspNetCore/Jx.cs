using System.Text.Json;

namespace Sharplight.AspNetCore;

/// <summary>Typed accessors over a request JSON object. Wrongly typed fields raise a 400 <see cref="ApiException"/>.</summary>
internal static class Jx
{
    extension(JsonElement o)
    {
        /// <summary>The property <paramref name="name"/>, or null when absent or JSON null (also null when the element is not an object).</summary>
        public JsonElement? Get(string name)
            => o.ValueKind == JsonValueKind.Object && o.TryGetProperty(name, out var v) && v.ValueKind != JsonValueKind.Null ? v : null;

        /// <summary>A string property, or null when absent or not a string.</summary>
        public string? Str(string name)
            => o.Get(name) is { ValueKind: JsonValueKind.String } v ? v.GetString() : null;

        /// <summary>An integer property; fractional numbers are truncated. <paramref name="param"/> names the field in the error (default: the property name).</summary>
        public int? Int(string name, string? param = null)
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

        /// <summary>A numeric property as float; null when absent.</summary>
        public float? Float(string name)
        {
            if (o.Get(name) is not { } v)
            {
                return null;
            }

            return v.ValueKind == JsonValueKind.Number ? v.GetSingle() : throw ApiException.BadRequest($"'{name}' must be a number.", name);
        }

        /// <summary>A boolean property; null when absent.</summary>
        public bool? Bool(string name)
        {
            if (o.Get(name) is not { } v)
            {
                return null;
            }

            return v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : throw ApiException.BadRequest($"'{name}' must be a boolean.", name);
        }

        /// <summary>The elements of an array property; empty when absent.</summary>
        public IEnumerable<JsonElement> Arr(string name)
        {
            if (o.Get(name) is not { } v)
            {
                return [];
            }

            return v.ValueKind == JsonValueKind.Array ? v.EnumerateArray() : throw ApiException.BadRequest($"'{name}' must be an array.", name);
        }

        /// <summary>An object property, or null when absent or not an object.</summary>
        public JsonElement? Obj(string name)
            => o.Get(name) is { ValueKind: JsonValueKind.Object } v ? v : null;
    }
}
