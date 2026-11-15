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
            ApplyChrome(window, _host.Services.Settings);
            var viewModel = new MainWindowViewModel(_host.Services);
            window.DataContext = viewModel;
            desktop.MainWindow = window;
            desktop.ShutdownRequested += (_, _) =>
            {
                viewModel.Updates.Dispose();
                _host.Dispose();
            };
            _ = viewModel.StartAsync();
        }

        base.OnFrameworkInitializationCompleted();
    }

    /// <summary>Frames the window with the own or the system title bar, and again whenever the setting changes.</summary>
    private static void ApplyChrome(MainWindow window, Services.Settings.ISettingsService settings)
    {
        var windowManagerMovesWindows = new Lazy<bool>(X11WindowManager.SupportsMoveResize);
        WindowChromeMode Choose() => WindowChrome.Choose(settings.Current.UseSystemTitleBar,
            Environment.GetEnvironmentVariable(WindowChrome.EnvironmentVariable), OperatingSystem.IsLinux(), () => windowManagerMovesWindows.Value);

        window.ApplyChrome(Choose());
        settings.Changed += (_, _) =>
        {
            var mode = Choose();
            if (mode != window.Chrome)
            {
                window.ApplyChrome(mode);
            }
        };
    }
}
