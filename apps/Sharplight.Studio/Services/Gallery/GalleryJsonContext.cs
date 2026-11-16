using System.Text.Json.Serialization;

namespace Sharplight.Studio.Services.Gallery;

/// <summary>Source-generated JSON for the gallery sidecars.</summary>
[JsonSourceGenerationOptions(WriteIndented = true, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(GalleryEntry))]
internal sealed partial class GalleryJsonContext : JsonSerializerContext;
