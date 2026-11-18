using System.CommandLine;
using Avalonia;
using Sharplight.Storage;
using Sharplight.Studio.Services.Settings;
using Velopack;

namespace Sharplight.Studio;

/// <summary>Entry point: <c>sharplight-studio [--models-dir DIR]</c>.</summary>
internal static class Program
{
    /// <summary>The model store given on the command line, or null.</summary>
    public static string? ModelsDirectory { get; private set; }

    [STAThread]
    public static int Main(string[] args)
    {
        // Velopack's install, update and uninstall hooks run (and exit) here, before anything else touches the process.
        VelopackApp.Build().Run();

        var modelsDir = new Option<string?>("--models-dir")
        {
            Description = "Model store directory (overrides the settings and SHARPLIGHT_MODELS, like the command-line tools).",
            HelpName = "dir",
        };
        var printStorage = new Option<bool>("--print-storage")
        {
            Description = "Print the resolved storage location (package folder, portable marker, config, data, models, cache, logs) and exit.",
        };
        var root = new RootCommand("Sharplight Studio: find, download, convert, chat with and serve local models.") { modelsDir, printStorage };
        var parsed = root.Parse(args);
        if (parsed.Errors.Count > 0 || args.Any(arg => arg is "--help" or "-h" or "-?" or "--version"))
        {
            return parsed.Invoke();
        }

        if (parsed.GetValue(printStorage))
        {
            Console.Out.Write(StorageReport.Create(StorageEnvironment.Process));
            return 0;
        }

        ModelsDirectory = parsed.GetValue(modelsDir);
        return BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }

    /// <summary>Avalonia configuration; also used by the designer and the headless tests.</summary>
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>().UsePlatformDetect().WithInterFont().LogToTrace()
        // The X11 window class matches StartupWMClass of packaging/linux/sharplight-studio.desktop, so the taskbar groups the window
        // under the launcher and shows its icon (also under XWayland).
        .With(new X11PlatformOptions { WmClass = WindowClass });

    /// <summary>The window class (X11 <c>WM_CLASS</c>) and desktop-file name of the app.</summary>
    public const string WindowClass = "sharplight-studio";
}
