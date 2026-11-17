using System.Text.Json.Serialization;

namespace Sharplight.Hub.Catalog;

/// <summary>Source-generated JSON metadata for <see cref="VerifiedCatalog"/>.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    WriteIndented = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    IgnoreReadOnlyProperties = true,
    ReadCommentHandling = System.Text.Json.JsonCommentHandling.Skip,
    AllowTrailingCommas = true)]
[JsonSerializable(typeof(VerifiedCatalog))]
internal sealed partial class CatalogJsonContext : JsonSerializerContext
{
}
