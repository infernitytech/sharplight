using System.Text.Json.Serialization;

namespace Sharplight.Architectures;

/// <summary>Source-generated JSON metadata of the Kokoro attachments.</summary>
[JsonSerializable(typeof(Dictionary<string, int>))]
internal sealed partial class KokoroJsonContext : JsonSerializerContext;
