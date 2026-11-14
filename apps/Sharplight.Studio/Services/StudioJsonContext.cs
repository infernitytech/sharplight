using System.Text.Json.Serialization;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services;

/// <summary>Source-generated JSON for the settings file and the stored conversations.</summary>
[JsonSourceGenerationOptions(WriteIndented = true, UseStringEnumConverter = true, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(AppSettings))]
[JsonSerializable(typeof(Conversation))]
internal sealed partial class StudioJsonContext : JsonSerializerContext;
