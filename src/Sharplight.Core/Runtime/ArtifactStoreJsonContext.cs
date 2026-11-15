using System.Text.Json.Serialization;

namespace Sharplight.Runtime;

/// <summary>Source-generated JSON metadata of <see cref="ArtifactStore"/> descriptor files (trimming and AOT safe).</summary>
[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(Dictionary<string, string>))]
internal sealed partial class ArtifactStoreJsonContext : JsonSerializerContext
{
}
