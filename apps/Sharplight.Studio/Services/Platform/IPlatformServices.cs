namespace Sharplight.Studio.Services.Platform;

/// <summary>Operating-system services the view models need, behind an interface so they can be tested without a window.</summary>
internal interface IPlatformServices
{
    /// <summary>Lets the user pick files; empty when cancelled.</summary>
    public Task<IReadOnlyList<string>> PickFilesAsync(string title, IReadOnlyList<FileFilter> filters, bool allowMultiple);

    /// <summary>Lets the user pick a folder; null when cancelled.</summary>
    public Task<string?> PickFolderAsync(string title);

    /// <summary>Lets the user choose where to save a file; null when cancelled.</summary>
    public Task<string?> PickSaveFileAsync(string title, string suggestedName, FileFilter filter);

    /// <summary>Copies text to the clipboard.</summary>
    public Task CopyTextAsync(string text);

    /// <summary>Opens a folder in the file manager or a URL in the browser.</summary>
    public void Open(string pathOrUrl);
}
