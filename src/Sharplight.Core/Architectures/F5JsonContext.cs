using System.Text.Json.Serialization;

namespace Sharplight.Architectures;

/// <summary>Source-generated JSON metadata of the F5-TTS attachments read by <see cref="F5Synthesizer"/>.</summary>
[JsonSerializable(typeof(string[]))]
internal sealed partial class F5JsonContext : JsonSerializerContext;
