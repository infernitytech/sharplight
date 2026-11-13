using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Sharplight.PackageViewer.ViewModels;
using Sharplight.PackageViewer.Views;

namespace Sharplight.PackageViewer;

/// <summary>The application: Fluent theme following the OS light/dark setting, one main window.</summary>
public sealed partial class App : Application
{
    /// <inheritdoc/>
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    /// <inheritdoc/>
    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var viewModel = new MainWindowViewModel();
            desktop.MainWindow = new MainWindow { DataContext = viewModel };
            if (desktop.Args is [var path, ..])
            {
                _ = viewModel.OpenAsync(path);
            }
        }

        base.OnFrameworkInitializationCompleted();
    }
}
