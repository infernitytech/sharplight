using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Services;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.Views;

namespace Sharplight.Studio;

/// <summary>The application: builds the services (<see cref="StudioHost"/>), applies the theme and shows the main window.</summary>
internal sealed partial class App : Application
{
    private StudioHost? _host;

    /// <inheritdoc/>
    public override void Initialize() => AvaloniaXamlLoader.Load(this);

    /// <inheritdoc/>
    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            var window = new MainWindow();
            var theme = new ThemeService(this);
            _host = StudioHost.Create(Program.ModelsDirectory, new AvaloniaPlatformServices(() => TopLevel.GetTopLevel(window)), new AvaloniaDispatcher(), theme);
            var settings = _host.Services.Settings.Current;
            theme.Apply(settings.Theme, settings.Accent);
            var viewModel = new MainWindowViewModel(_host.Services);
            window.DataContext = viewModel;
            desktop.MainWindow = window;
            desktop.ShutdownRequested += (_, _) => _host.Dispose();
            _ = viewModel.StartAsync();
        }

        base.OnFrameworkInitializationCompleted();
    }
}
