using System.Text.Json;
using System.Text.Json.Serialization;

namespace Sharplight.Serve;

/// <summary>Reads a device from JSON as either a number (<c>0</c>) or a string (<c>"0"</c>, <c>"cpu"</c>).</summary>
internal sealed class DeviceNameConverter : JsonConverter<string>
{
    public override string? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        return reader.TokenType == JsonTokenType.Number
            ? reader.GetInt32().ToString(System.Globalization.CultureInfo.InvariantCulture)
            : reader.GetString();
    }

    public override void Write(Utf8JsonWriter writer, string value, JsonSerializerOptions options)
    {
        writer.WriteStringValue(value);
    }
}
