using System.Text.Json;
using System.Text.Json.Serialization;

namespace Sharplight.Runtime;

/// <summary>Source-generated JSON metadata of <see cref="DeviceProfile"/> (trimming and AOT safe).</summary>
[JsonSourceGenerationOptions(
    WriteIndented = true,
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    PropertyNameCaseInsensitive = true,
    UseStringEnumConverter = true,
    ReadCommentHandling = JsonCommentHandling.Skip,
    AllowTrailingCommas = true)]
[JsonSerializable(typeof(DeviceProfile))]
internal sealed partial class DeviceProfileJsonContext : JsonSerializerContext
{
}
