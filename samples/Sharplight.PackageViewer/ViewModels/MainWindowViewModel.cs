using System.Diagnostics;
using CommunityToolkit.Mvvm.ComponentModel;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>The main window: the opened package (if any), the window title and the status line.</summary>
public sealed partial class MainWindowViewModel : ObservableObject
{
    public const string AppTitle = "Sharplight Package Viewer";

    [ObservableProperty]
    public partial PackageViewModel? Package { get; private set; }

    /// <summary>The application name, prefixed with the open file's name.</summary>
    [ObservableProperty]
    public partial string Title { get; private set; } = AppTitle;

    [ObservableProperty]
    public partial string Status { get; private set; } = "Open a .slm package: Open..., drag and drop, or pass the path on the command line.";

    [ObservableProperty]
    public partial bool IsLoading { get; private set; }

    /// <summary>Opens a package (reading happens on a background thread). Errors are shown in the status line. Call on the UI thread.</summary>
    /// <returns>Whether the package was opened.</returns>
    public async Task<bool> OpenAsync(string path)
    {
        IsLoading = true;
        Status = $"Opening {path}...";
        var watch = Stopwatch.StartNew();
        try
        {
            var package = await Task.Run(() => PackageViewModel.Load(path));
            Package = package;
            package.Graphs.SelectedGraph = package.Graphs.Graphs.FirstOrDefault();
            Title = $"{package.FileName} - {AppTitle}";
            Status = $"{package.Path} ({watch.ElapsedMilliseconds} ms)";
            return true;
        }
        catch (Exception e) when (e is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            Status = $"Cannot open {path}: {e.Message}";
            return false;
        }
        finally
        {
            IsLoading = false;
        }
    }
}
