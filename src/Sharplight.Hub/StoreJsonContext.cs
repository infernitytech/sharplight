using System.Text.Json.Serialization;

namespace Sharplight.Hub;

/// <summary>Source-generated JSON metadata for <see cref="ModelManifest"/>.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    UseStringEnumConverter = true,
    WriteIndented = true,
    PropertyNameCaseInsensitive = true)]
[JsonSerializable(typeof(ModelManifest))]
internal sealed partial class StoreJsonContext : JsonSerializerContext
{
}
