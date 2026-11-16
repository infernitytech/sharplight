using System.Text.Json;
using Sharplight.Storage;

namespace Sharplight.Hub;

/// <summary>
/// The configured model store location. The command-line tools read it from a small shared file in the configuration folder of the
/// storage layout (<see cref="FileFor"/>); applications with their own settings (Sharplight Studio) keep the same choice there.
/// </summary>
public sealed record ModelStoreSettings
{
    /// <summary>File name of the shared settings file.</summary>
    public const string FileName = "store.json";

    private const string TemporarySuffix = ".tmp";

    /// <summary>Where the store lives.</summary>
    public ModelStoreLocation Location { get; init; } = ModelStoreLocation.Application;

    /// <summary>The directory for <see cref="ModelStoreLocation.Custom"/>; ignored otherwise.</summary>
    public string? CustomPath { get; init; }

    /// <summary>The shared settings file of the tools: <c>store.json</c> in the configuration folder of <paramref name="layout"/>.</summary>
    /// <param name="layout">The storage layout in effect.</param>
    public static string FileFor(StorageLayout layout)
    {
        ArgumentNullException.ThrowIfNull(layout);
        return Path.Combine(layout.ConfigDirectory, FileName);
    }

    /// <summary>Reads a settings file.</summary>
    /// <param name="path">The file.</param>
    /// <returns>The settings, or <see langword="null"/> when the file does not exist or cannot be read.</returns>
    public static ModelStoreSettings? Load(string path)
    {
        try
        {
            return File.Exists(path) ? JsonSerializer.Deserialize(File.ReadAllText(path), StoreJsonContext.Default.ModelStoreSettings) : null;
        }
        catch (Exception error) when (error is JsonException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>Writes the settings to <paramref name="path"/> atomically, creating its directory.</summary>
    /// <param name="path">The file.</param>
    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        var temporary = path + TemporarySuffix;
        File.WriteAllText(temporary, JsonSerializer.Serialize(this, StoreJsonContext.Default.ModelStoreSettings));
        File.Move(temporary, path, overwrite: true);
    }
}
