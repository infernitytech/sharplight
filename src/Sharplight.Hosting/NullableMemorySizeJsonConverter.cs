using System.Text.Json.Serialization;
using System.Text.Json;

namespace Sharplight.Hosting;

/// <summary><see cref="MemorySizeJsonConverter"/> for optional sizes (<c>null</c> stays null).</summary>
internal sealed class NullableMemorySizeJsonConverter : JsonConverter<long?>
{
    private static readonly MemorySizeJsonConverter s_inner = new();

    public override bool HandleNull => true;

    public override long? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        return reader.TokenType == JsonTokenType.Null ? null : s_inner.Read(ref reader, typeof(long), options);
    }

    public override void Write(Utf8JsonWriter writer, long? value, JsonSerializerOptions options)
    {
        if (value is { } bytes)
        {
            writer.WriteNumberValue(bytes);
            return;
        }

        writer.WriteNullValue();
    }
}
