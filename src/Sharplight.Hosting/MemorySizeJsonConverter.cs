using System.Text.Json.Serialization;
using System.Text.Json;

namespace Sharplight.Hosting;

/// <summary>Reads a byte size as a number of bytes or a string with a unit suffix (<see cref="MemorySize.Parse"/>); writes the number.</summary>
internal sealed class MemorySizeJsonConverter : JsonConverter<long>
{
    public override long Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.Number)
        {
            return reader.GetInt64();
        }

        try
        {
            return MemorySize.Parse(reader.GetString() ?? "");
        }
        catch (FormatException error)
        {
            throw new JsonException(error.Message, error);
        }
    }

    public override void Write(Utf8JsonWriter writer, long value, JsonSerializerOptions options)
    {
        writer.WriteNumberValue(value);
    }
}
