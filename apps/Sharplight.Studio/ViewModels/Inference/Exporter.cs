using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>Export for every inference page: asks where to save and writes the content (or copies an output file there).</summary>
internal static class Exporter
{
    /// <summary>Saves <paramref name="content"/> where the user chooses.</summary>
    /// <param name="platform">The file dialog.</param>
    /// <param name="title">The dialog's title.</param>
    /// <param name="suggestedName">The file name offered.</param>
    /// <param name="content">The bytes to write.</param>
    /// <returns>The file written, or null when the user cancelled.</returns>
    public static async Task<string?> SaveAsync(IPlatformServices platform, string title, string suggestedName, byte[] content)
    {
        string extension = Path.GetExtension(suggestedName);
        var filter = new FileFilter(extension.Length > 1 ? extension[1..].ToUpperInvariant() + " file" : "File", [extension.Length > 1 ? "*" + extension : "*"]);
        if (await platform.PickSaveFileAsync(title, suggestedName, filter) is not { } path)
        {
            return null;
        }

        await File.WriteAllBytesAsync(path, content);
        return path;
    }

    /// <summary>Saves text (UTF-8) where the user chooses.</summary>
    /// <param name="platform">The file dialog.</param>
    /// <param name="title">The dialog's title.</param>
    /// <param name="suggestedName">The file name offered.</param>
    /// <param name="text">The text to write.</param>
    /// <returns>The file written, or null when the user cancelled.</returns>
    public static Task<string?> SaveTextAsync(IPlatformServices platform, string title, string suggestedName, string text)
        => SaveAsync(platform, title, suggestedName, new System.Text.UTF8Encoding(false).GetBytes(text));

    /// <summary>Copies an output file where the user chooses.</summary>
    /// <param name="platform">The file dialog.</param>
    /// <param name="source">The output file.</param>
    /// <returns>The file written, or null when the user cancelled.</returns>
    public static async Task<string?> CopyAsync(IPlatformServices platform, string source)
        => await SaveAsync(platform, "Export", Path.GetFileName(source), await File.ReadAllBytesAsync(source));
}
