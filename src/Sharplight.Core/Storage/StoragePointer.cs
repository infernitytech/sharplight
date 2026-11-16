using System.Text.Json;

namespace Sharplight.Storage;

/// <summary>
/// The user's storage choice when it is not portable: a small JSON file in the system configuration folder
/// (<see cref="StorageLayout.PointerFile"/>), read before anything else so every Sharplight tool of the user finds the same
/// folders. A portable package ignores it (its marker file wins).
/// </summary>
public sealed record StoragePointer
{
    /// <summary>File name of the pointer file.</summary>
    public const string FileName = "storage.json";

    /// <summary>The chosen mode: <see cref="StorageMode.System"/> or <see cref="StorageMode.Custom"/>.</summary>
    public StorageMode Mode { get; init; } = StorageMode.System;

    /// <summary>The root folder of <see cref="StorageMode.Custom"/>; ignored otherwise.</summary>
    public string? Root { get; init; }

    /// <summary>Reads a pointer file.</summary>
    /// <param name="path">The file.</param>
    /// <returns>The pointer, or <see langword="null"/> when the file does not exist or cannot be read.</returns>
    public static StoragePointer? Load(string path)
    {
        try
        {
            return File.Exists(path) ? JsonSerializer.Deserialize(File.ReadAllText(path), StorageJsonContext.Default.StoragePointer) : null;
        }
        catch (Exception error) when (error is JsonException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>Writes the pointer to <paramref name="path"/> atomically, creating its folder.</summary>
    /// <param name="path">The file.</param>
    public void Save(string path)
    {
        StorageFiles.WriteAtomically(path, JsonSerializer.Serialize(this, StorageJsonContext.Default.StoragePointer));
    }
}
