using System.Text.Json.Serialization;

namespace Sharplight.Import.F5Tts;

/// <summary>Source-generated JSON metadata of the F5-TTS importer's attachments.</summary>
[JsonSerializable(typeof(string[]))]
internal sealed partial class F5TtsJsonContext : JsonSerializerContext;
