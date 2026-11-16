using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Services.Platform;

namespace Sharplight.Studio.ViewModels;

/// <summary>The About dialog: name, version (from the build's version number), license and links.</summary>
internal sealed partial class AboutViewModel : ObservableObject
{
    /// <summary>The application's name, as the menus and the window show it.</summary>
    public const string ApplicationName = "Sharplight Studio";

    /// <summary>The repository.</summary>
    public const string RepositoryUrl = "https://github.com/infernitytech/sharplight";

    private readonly IPlatformServices _platform;

    /// <param name="platform">Opens the links.</param>
    /// <param name="version">The version the app runs (the update service's current version).</param>
    public AboutViewModel(IPlatformServices platform, string version)
    {
        _platform = platform;
        Version = version;
    }

    /// <summary>The application's name.</summary>
    public string Name => ApplicationName;

    /// <summary>The version.</summary>
    public string Version { get; }

    /// <summary>One line about what the app is.</summary>
    public string Description => "Find, download, convert, run and serve local AI models, on the Sharplight inference engine for .NET.";

    /// <summary>The license.</summary>
    public string License => "MIT License. MoltenVK (macOS): Apache License 2.0.";

    /// <summary>The copyright line.</summary>
    public string Copyright => "Copyright (c) Sharplight contributors.";

    /// <summary>The repository.</summary>
    public string Repository => RepositoryUrl;

    /// <summary>The dialog is shown.</summary>
    [ObservableProperty]
    public partial bool IsOpen { get; set; }

    /// <summary>Opens a link in the browser.</summary>
    [RelayCommand]
    private void OpenLink(string? url)
    {
        if (!string.IsNullOrEmpty(url))
        {
            _platform.Open(url);
        }
    }

    /// <summary>Closes the dialog.</summary>
    [RelayCommand]
    private void Close() => IsOpen = false;
}
