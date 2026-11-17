using System.Text.Json;
using System.Text.Json.Serialization;
using Sharplight.ModelFeed.Client;
using Sharplight.ModelFeed.Index;
using Sharplight.ModelFeed.Signing;

namespace Sharplight.ModelFeed;

/// <summary>Source-generated JSON metadata of the feed documents (camelCase, nulls omitted, comments and trailing commas accepted).</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    WriteIndented = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    IgnoreReadOnlyProperties = true,
    ReadCommentHandling = JsonCommentHandling.Skip,
    AllowTrailingCommas = true)]
[JsonSerializable(typeof(PackageManifest))]
[JsonSerializable(typeof(FeedServiceIndex))]
[JsonSerializable(typeof(FeedCatalogIndex))]
[JsonSerializable(typeof(FeedCatalogPage))]
[JsonSerializable(typeof(FeedPackageIndex))]
[JsonSerializable(typeof(FeedKeySet))]
[JsonSerializable(typeof(FeedKeyFile))]
[JsonSerializable(typeof(FeedState))]
[JsonSerializable(typeof(Dictionary<string, FeedState>))]
[JsonSerializable(typeof(JwsHeader))]
public sealed partial class FeedJsonContext : JsonSerializerContext
{
}
