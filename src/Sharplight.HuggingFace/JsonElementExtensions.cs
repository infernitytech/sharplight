using System.Text.Json;

namespace Sharplight.HuggingFace;

/// <summary>Shorthands for the lenient reads tokenizer and config JSON files need (absent or mistyped properties count as "no").</summary>
internal static class JsonElementExtensions
{
    extension(JsonElement element)
    {
        /// <summary>Whether <paramref name="property"/> exists and is the JSON literal <c>true</c>.</summary>
        public bool IsTrue(string property)
            => element.TryGetProperty(property, out var value) && value.ValueKind == JsonValueKind.True;

        /// <summary>The element's <c>type</c> property as a string; empty when absent.</summary>
        public string TypeName() => element.TryGetProperty("type", out var type) ? type.GetString() ?? "" : "";

        /// <summary>The elements of a JSON array property, or an empty array when absent or not an array.</summary>
        public JsonElement[] ArrayOrEmpty(string property)
            => element.TryGetProperty(property, out var value) && value.ValueKind == JsonValueKind.Array ? value.EnumerateArray().ToArray() : [];
    }
}
