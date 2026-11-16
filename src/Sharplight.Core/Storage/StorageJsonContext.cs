using System.Text.Json.Serialization;

namespace Sharplight.Storage;

/// <summary>Source-generated JSON metadata of the storage files (trimming and AOT safe).</summary>
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    UseStringEnumConverter = true,
    WriteIndented = true,
    PropertyNameCaseInsensitive = true)]
[JsonSerializable(typeof(StoragePointer))]
[JsonSerializable(typeof(StorageMigrationJournal))]
internal sealed partial class StorageJsonContext : JsonSerializerContext
{
}
