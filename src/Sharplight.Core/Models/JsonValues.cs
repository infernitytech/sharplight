using System.Text.Json;

namespace Sharplight.Models;

/// <summary>Reflection-free conversion of JSON configuration values (<see cref="ModelComponent.Config"/>) to CLR scalars.</summary>
internal static class JsonValues
{
    /// <summary>Converts a JSON value to <typeparamref name="T"/> (string, bool, int, long, float, double, or <see cref="JsonElement"/>).</summary>
    /// <exception cref="FormatException">The value cannot be converted.</exception>
    public static T Convert<T>(JsonElement value)
    {
        object converted = typeof(T) switch
        {
            var t when t == typeof(JsonElement) => value,
            var t when t == typeof(string) => value.ValueKind == JsonValueKind.String ? value.GetString()! : value.GetRawText(),
            var t when t == typeof(bool) => value.ValueKind switch
            {
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                _ => throw Mismatch(value, typeof(T)),
            },
            var t when t == typeof(int) => value.ValueKind == JsonValueKind.Number ? value.GetInt32() : throw Mismatch(value, typeof(T)),
            var t when t == typeof(long) => value.ValueKind == JsonValueKind.Number ? value.GetInt64() : throw Mismatch(value, typeof(T)),
            var t when t == typeof(float) => value.ValueKind == JsonValueKind.Number ? value.GetSingle() : throw Mismatch(value, typeof(T)),
            var t when t == typeof(double) => value.ValueKind == JsonValueKind.Number ? value.GetDouble() : throw Mismatch(value, typeof(T)),
            _ => throw new FormatException($"Configuration values cannot be read as {typeof(T).Name}."),
        };
        return (T)converted;
    }

    private static FormatException Mismatch(JsonElement value, Type type)
        => new($"The configuration value {value.GetRawText()} is not a {type.Name}.");
}
