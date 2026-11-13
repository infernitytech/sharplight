using Avalonia;

namespace Sharplight.PackageViewer;

/// <summary>Entry point: <c>sharplight-viewer [package.slm]</c>.</summary>
internal static class Program
{
    [STAThread]
    public static int Main(string[] args) => BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);

    /// <summary>Avalonia configuration; also used by the designer.</summary>
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>().UsePlatformDetect().LogToTrace();
}
