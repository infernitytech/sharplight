using System.Text.Json.Serialization;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services;

/// <summary>Source-generated JSON for the settings file, the stored conversations and the cached Discover highlights.</summary>
[JsonSourceGenerationOptions(WriteIndented = true, UseStringEnumConverter = true, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(AppSettings))]
[JsonSerializable(typeof(Conversation))]
[JsonSerializable(typeof(Hub.HighlightsCacheFile))]
internal sealed partial class StudioJsonContext : JsonSerializerContext;
