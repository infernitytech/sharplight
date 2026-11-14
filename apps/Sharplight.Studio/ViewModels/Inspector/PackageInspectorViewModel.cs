using System.Diagnostics;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Hub;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>
/// The package inspector: everything stored in one <c>.slm</c> file (overview, graphs with a visual graph view and the IR, weights,
/// precompiled artifacts, metadata and the raw section layout). Opened from the model library, or for any package file.
/// </summary>
internal sealed partial class PackageInspectorViewModel : ObservableObject
{
    private readonly ModelManifest? _manifest;
    private readonly Action _close;

    /// <param name="path">The package file.</param>
    /// <param name="manifest">The store record when opened from the library, else null.</param>
    /// <param name="close">Closes the inspector (back to the library).</param>
    public PackageInspectorViewModel(string path, ModelManifest? manifest, Action close)
    {
        Path = path;
        _manifest = manifest;
        _close = close;
        Title = manifest?.Name ?? System.IO.Path.GetFileName(path);
        Status = $"Reading {path}…";
    }

    /// <summary>The package file.</summary>
    public string Path { get; }

    /// <summary>Model or file name.</summary>
    [ObservableProperty]
    public partial string Title { get; private set; }

    /// <summary>Size, version and read time, or the error.</summary>
    [ObservableProperty]
    public partial string Status { get; private set; }

    /// <summary>True while the package is read.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsLoaded))]
    public partial bool IsLoading { get; private set; }

    /// <summary>Why the file could not be read, or null.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasError))]
    public partial string? Error { get; private set; }

    /// <summary>True when the file could not be read.</summary>
    public bool HasError => Error is not null;

    /// <summary>The tab view models, once read.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsLoaded))]
    public partial PackageInspection? Package { get; private set; }

    /// <summary>True when the package was read.</summary>
    public bool IsLoaded => Package is not null && !IsLoading;

    /// <summary>The visible tab (<see cref="InspectorTab"/>); 0 until the package was read.</summary>
    public int SelectedTab
    {
        get => Package?.SelectedTab ?? 0;
        set
        {
            if (Package is { } package)
            {
                package.SelectedTab = value;
            }
        }
    }

    /// <summary>Reads the package on a background thread and selects the first graph. Call on the UI thread.</summary>
    /// <returns>Whether the package was read.</returns>
    public async Task<bool> LoadAsync()
    {
        IsLoading = true;
        Error = null;
        var watch = Stopwatch.StartNew();
        try
        {
            var path = Path;
            var manifest = _manifest;
            var package = await Task.Run(() => PackageInspection.Read(path, manifest));
            Package = package;
            Title = _manifest?.Name ?? package.Overview.Title;
            package.Graphs.SelectedGraph = package.Graphs.Graphs.FirstOrDefault(g => g.Name == Sharplight.Models.WellKnown.DecodeGraph)
                ?? package.Graphs.Graphs.FirstOrDefault();
            Status = $"{ByteFormat.Format(package.Info.FileSize)} · format version {package.Info.Version} · read in {watch.ElapsedMilliseconds} ms";
            return true;
        }
        catch (Exception e) when (e is IOException or InvalidDataException or UnauthorizedAccessException or NotSupportedException)
        {
            Error = $"Cannot read {Path}: {e.Message}";
            Status = "Not readable";
            return false;
        }
        finally
        {
            IsLoading = false;
        }
    }

    /// <summary>Shows the tab that decodes a file region (for an artifact set, selects it).</summary>
    public void ShowSection(SectionRow? row) => Package?.ShowSection(row);

    [RelayCommand]
    private void Close() => _close();
}
