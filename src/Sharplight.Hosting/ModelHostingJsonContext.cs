using System.Text.Json.Serialization;
using System.Text.Json;

namespace Sharplight.Hosting;

/// <summary>Source-generated JSON metadata of the hosting options file.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    PropertyNameCaseInsensitive = true,
    ReadCommentHandling = JsonCommentHandling.Skip,
    AllowTrailingCommas = true,
    WriteIndented = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(ModelHostingOptions))]
internal sealed partial class ModelHostingJsonContext : JsonSerializerContext
{
}
