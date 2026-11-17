using System.Text.Json.Serialization;

namespace Sharplight.Speech;

/// <summary>Source-generated JSON metadata of the stored voices (trimming and AOT safe).</summary>
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, WriteIndented = true, PropertyNameCaseInsensitive = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(VoiceProfileManifest))]
internal sealed partial class SpeechJsonContext : JsonSerializerContext
{
}
