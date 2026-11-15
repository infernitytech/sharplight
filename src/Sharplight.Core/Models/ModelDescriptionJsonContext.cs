using System.Text.Json;
using System.Text.Json.Serialization;

namespace Sharplight.Models;

/// <summary>Source-generated JSON metadata of <see cref="ModelDescription"/> (trimming and AOT safe).</summary>
[JsonSourceGenerationOptions(
    WriteIndented = true,
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    PropertyNameCaseInsensitive = true,
    ReadCommentHandling = JsonCommentHandling.Skip,
    AllowTrailingCommas = true)]
[JsonSerializable(typeof(ModelDescription))]
internal sealed partial class ModelDescriptionJsonContext : JsonSerializerContext
{
}
