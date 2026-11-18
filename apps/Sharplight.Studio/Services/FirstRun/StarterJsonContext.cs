using System.Text.Json.Serialization;

namespace Sharplight.Studio.Services.FirstRun;

/// <summary>Source-generated JSON of the starter preferences (camelCase, enum names, comments allowed).</summary>
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, PropertyNameCaseInsensitive = true, UseStringEnumConverter = true,
    ReadCommentHandling = System.Text.Json.JsonCommentHandling.Skip, AllowTrailingCommas = true)]
[JsonSerializable(typeof(StarterPreferences))]
internal sealed partial class StarterJsonContext : JsonSerializerContext;
