using System.Text.Json.Serialization;

namespace Sharplight.Hosting;

/// <summary>Source-generated JSON metadata of <see cref="HostStatus"/> (camelCase names, enums as strings).</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    UseStringEnumConverter = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(HostStatus))]
internal sealed partial class HostStatusJsonContext : JsonSerializerContext
{
}
