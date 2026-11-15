using System.Text.Json.Serialization;

namespace Sharplight.Tokenizers;

/// <summary>Source-generated JSON metadata of <see cref="TokenizerData"/> (trimming and AOT safe).</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    PropertyNameCaseInsensitive = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(TokenizerData))]
internal sealed partial class TokenizerDataJsonContext : JsonSerializerContext
{
}
