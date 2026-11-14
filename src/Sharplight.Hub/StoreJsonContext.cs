using System.Text.Json.Serialization;

namespace Sharplight.Hub;

/// <summary>Source-generated JSON metadata for <see cref="ModelManifest"/> and <see cref="ModelStoreSettings"/>.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    UseStringEnumConverter = true,
    WriteIndented = true,
    PropertyNameCaseInsensitive = true)]
[JsonSerializable(typeof(ModelManifest))]
[JsonSerializable(typeof(ModelStoreSettings))]
internal sealed partial class StoreJsonContext : JsonSerializerContext
{
}
