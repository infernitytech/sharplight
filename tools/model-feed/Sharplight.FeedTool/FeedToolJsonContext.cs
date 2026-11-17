using System.Text.Json;
using System.Text.Json.Serialization;

namespace Sharplight.FeedTool;

/// <summary>JSON metadata of the overlay file.</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    ReadCommentHandling = JsonCommentHandling.Skip,
    AllowTrailingCommas = true)]
[JsonSerializable(typeof(FeedOverlay))]
internal sealed partial class FeedToolJsonContext : JsonSerializerContext
{
}
