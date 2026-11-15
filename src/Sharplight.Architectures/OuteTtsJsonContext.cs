using System.Text.Json.Serialization;

namespace Sharplight.Architectures;

/// <summary>Source-generated JSON metadata of the OuteTTS speaker profile.</summary>
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(OuteTtsVoice))]
internal sealed partial class OuteTtsJsonContext : JsonSerializerContext;
