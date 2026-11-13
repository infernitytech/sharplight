using System.Text.Json;
using System.Text.Json.Serialization;

namespace Sharplight.Server;

/// <summary>Source-generated JSON metadata for the configuration file.</summary>
[JsonSourceGenerationOptions(
    PropertyNameCaseInsensitive = true,
    ReadCommentHandling = JsonCommentHandling.Skip,
    AllowTrailingCommas = true)]
[JsonSerializable(typeof(ServerConfig))]
internal sealed partial class ServeJsonContext : JsonSerializerContext
{
}
