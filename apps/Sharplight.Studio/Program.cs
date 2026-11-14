using System.CommandLine;
using Avalonia;

namespace Sharplight.Studio;

/// <summary>Entry point: <c>sharplight-studio [--models-dir DIR]</c>.</summary>
internal static class Program
{
    /// <summary>The model store given on the command line, or null.</summary>
    public static string? ModelsDirectory { get; private set; }

    [STAThread]
    public static int Main(string[] args)
    {
        var modelsDir = new Option<string?>("--models-dir")
        {
            Description = "Model store directory (overrides the settings and SHARPLIGHT_MODELS, like the command-line tools).",
            HelpName = "dir",
        };
        var root = new RootCommand("Sharplight Studio: find, download, convert, chat with and serve local models.") { modelsDir };
        var parsed = root.Parse(args);
        if (parsed.Errors.Count > 0 || args.Any(arg => arg is "--help" or "-h" or "-?" or "--version"))
        {
            return parsed.Invoke();
        }

        ModelsDirectory = parsed.GetValue(modelsDir);
        return BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }

    /// <summary>Avalonia configuration; also used by the designer and the headless tests.</summary>
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>().UsePlatformDetect().WithInterFont().LogToTrace();
}
