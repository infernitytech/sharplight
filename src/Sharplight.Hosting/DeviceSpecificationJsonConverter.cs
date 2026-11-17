using System.Globalization;
using System.Text.Json.Serialization;
using System.Text.Json;

namespace Sharplight.Hosting;

/// <summary>Reads a device specification as a number (<c>0</c>) or a string (<c>"0"</c>, <c>"cpu"</c>, <c>"vulkan:1"</c>, <c>"0,1"</c>).</summary>
internal sealed class DeviceSpecificationJsonConverter : JsonConverter<string>
{
    public override string? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        return reader.TokenType == JsonTokenType.Number
            ? reader.GetInt32().ToString(CultureInfo.InvariantCulture)
            : reader.GetString();
    }

    public override void Write(Utf8JsonWriter writer, string value, JsonSerializerOptions options)
    {
        writer.WriteStringValue(value);
    }
}
