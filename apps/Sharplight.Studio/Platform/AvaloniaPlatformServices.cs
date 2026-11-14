using Avalonia.Controls;
using Avalonia.Input.Platform;
using Avalonia.Platform.Storage;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.Platform;

/// <summary><see cref="IPlatformServices"/> through the main window's storage provider, clipboard and launcher.</summary>
internal sealed class AvaloniaPlatformServices(Func<TopLevel?> topLevel) : IPlatformServices
{
    /// <inheritdoc/>
    public async Task<IReadOnlyList<string>> PickFilesAsync(string title, IReadOnlyList<FileFilter> filters, bool allowMultiple)
    {
        if (topLevel() is not { } top)
        {
            return [];
        }

        var files = await top.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = title,
            AllowMultiple = allowMultiple,
            FileTypeFilter = [.. filters.Select(filter => new FilePickerFileType(filter.Name) { Patterns = filter.Patterns })],
        });
        return [.. files.Select(file => file.TryGetLocalPath()).OfType<string>()];
    }

    /// <inheritdoc/>
    public async Task<string?> PickFolderAsync(string title)
    {
        if (topLevel() is not { } top)
        {
            return null;
        }

        var folders = await top.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions { Title = title });
        return folders.Count > 0 ? folders[0].TryGetLocalPath() : null;
    }

    /// <inheritdoc/>
    public async Task<string?> PickSaveFileAsync(string title, string suggestedName, FileFilter filter)
    {
        if (topLevel() is not { } top)
        {
            return null;
        }

        var file = await top.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = title,
            SuggestedFileName = suggestedName,
            FileTypeChoices = [new FilePickerFileType(filter.Name) { Patterns = filter.Patterns }],
        });
        return file?.TryGetLocalPath();
    }

    /// <inheritdoc/>
    public async Task CopyTextAsync(string text)
    {
        if (topLevel()?.Clipboard is { } clipboard)
        {
            await clipboard.SetTextAsync(text);
        }
    }

    /// <inheritdoc/>
    public void Open(string pathOrUrl)
    {
        if (topLevel()?.Launcher is not { } launcher)
        {
            return;
        }

        if (Uri.TryCreate(pathOrUrl, UriKind.Absolute, out var uri) && (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps))
        {
            _ = launcher.LaunchUriAsync(uri);
        }
        else if (Directory.Exists(pathOrUrl))
        {
            _ = launcher.LaunchDirectoryInfoAsync(new DirectoryInfo(pathOrUrl));
        }
        else if (File.Exists(pathOrUrl))
        {
            _ = launcher.LaunchFileInfoAsync(new FileInfo(pathOrUrl));
        }
    }
}
