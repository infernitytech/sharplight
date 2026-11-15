using System.Text.Json;

namespace Sharplight.Import.Diffusers;

/// <summary>Typed reads of diffusers / transformers configuration files, with defaults for absent or null entries.</summary>
internal static class JsonConfigExtensions
{
    extension(JsonElement config)
    {
        /// <summary>An integer entry, or <paramref name="defaultValue"/>.</summary>
        public int Int(string key, int defaultValue)
            => config.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetInt32() : defaultValue;

        /// <summary>A number entry, or <paramref name="defaultValue"/>.</summary>
        public float Float(string key, float defaultValue)
            => config.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetSingle() : defaultValue;

        /// <summary>A boolean entry, or <paramref name="defaultValue"/>.</summary>
        public bool Bool(string key, bool defaultValue)
            => config.TryGetProperty(key, out var value) && value.ValueKind is JsonValueKind.True or JsonValueKind.False ? value.GetBoolean() : defaultValue;

        /// <summary>A string entry, or <paramref name="defaultValue"/>.</summary>
        public string String(string key, string defaultValue)
            => config.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString()! : defaultValue;

        /// <summary>An integer array entry.</summary>
        /// <exception cref="InvalidDataException">The entry is missing.</exception>
        public int[] Ints(string key)
            => config.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Array
                ? [.. value.EnumerateArray().Select(e => e.GetInt32())]
                : throw new InvalidDataException($"The configuration has no '{key}' list.");

        /// <summary>A string array entry.</summary>
        /// <exception cref="InvalidDataException">The entry is missing.</exception>
        public string[] Strings(string key)
            => config.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Array
                ? [.. value.EnumerateArray().Select(e => e.GetString()!)]
                : throw new InvalidDataException($"The configuration has no '{key}' list.");

        /// <summary>An entry that is either one integer for all <paramref name="count"/> levels or a list of them.</summary>
        public int[] IntOrList(string key, int defaultValue, int count)
        {
            if (config.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Array)
            {
                return [.. value.EnumerateArray().Select(e => e.GetInt32())];
            }

            int single = config.Int(key, defaultValue);
            return [.. Enumerable.Repeat(single, count)];
        }
    }
}
