using System.CommandLine;
using System.Reflection;
using Avalonia;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Sharplight.Storage;
using Sharplight.Studio.Services.Logging;
using Sharplight.Studio.Services.Settings;
using Sharplight.Studio.Services.Smoke;
using Velopack;

namespace Sharplight.Studio;

/// <summary>Entry point: <c>sharplight-studio [--models-dir DIR]</c>.</summary>
internal static class Program
{
    /// <summary>The model store given on the command line, or null.</summary>
    public static string? ModelsDirectory { get; private set; }

    /// <summary>The logger factory writing to the log folder of the storage layout (a no-op factory until <see cref="Main"/> created it, as in tests).</summary>
    public static ILoggerFactory Logs { get; private set; } = NullLoggerFactory.Instance;

    /// <summary>The version of this build (the assembly's informational version without build metadata), as the update service reports it.</summary>
    public static string Version { get; } = typeof(Program).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion.Split('+')[0] ?? "dev";

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
        var smoke = new Option<bool>("--smoke")
        {
            Description = "Start the services without a window, install the Whisper tiny package from the Sharplight feed, transcribe a short WAV once on the CPU, write a JSON report and exit (0 passed, 1 failed).",
        };
        var smokeReport = new Option<string?>("--smoke-report")
        {
            Description = "With --smoke: where to write the JSON report (default: smoke-report.json in the log folder).",
            HelpName = "file",
        };
        var root = new RootCommand("Sharplight Studio: find, download, convert, chat with and serve local models.") { modelsDir, printStorage, smoke, smokeReport };
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
        var layout = StorageLayout.Resolve(StorageEnvironment.Process);
        Logs = StudioLogging.Create(layout, Environment.GetEnvironmentVariable(StudioLogging.LevelVariable));
        var log = Logs.CreateLogger("Sharplight.Studio");
        StudioLogging.Hook(log);
        log.LogInformation("Sharplight Studio {Version} starting on {Os} ({Architecture}); storage {Mode} at {Location}",
            typeof(Program).Assembly.GetName().Version, System.Runtime.InteropServices.RuntimeInformation.OSDescription,
            System.Runtime.InteropServices.RuntimeInformation.OSArchitecture, layout.Mode, layout.Description);
        if (parsed.GetValue(smoke))
        {
            return SmokeCommand.Run(layout, Logs, parsed.GetValue(smokeReport));
        }

        try
        {
            var code = BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
            log.LogInformation("Sharplight Studio exited with code {Code}", code);
            return code;
        }
        catch (Exception error)
        {
            log.LogCritical(error, "Sharplight Studio failed to start or crashed");
            throw;
        }
    }

    /// <summary>Avalonia configuration; also used by the designer and the headless tests.</summary>
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>().UsePlatformDetect().WithInterFont().LogToTrace()
        // The X11 window class matches StartupWMClass of packaging/linux/sharplight-studio.desktop, so the taskbar groups the window
        // under the launcher and shows its icon (also under XWayland).
        .With(new X11PlatformOptions { WmClass = WindowClass });

    /// <summary>The window class (X11 <c>WM_CLASS</c>) and desktop-file name of the app.</summary>
    public const string WindowClass = "sharplight-studio";
}
