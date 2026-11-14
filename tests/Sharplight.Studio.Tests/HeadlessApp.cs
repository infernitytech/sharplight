using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Threading;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The Studio <see cref="App"/> on Avalonia's headless platform with Skia rendering (so frames can be captured). Tests run their UI
/// code with <see cref="RunAsync"/> on the session's UI thread (one session for the test run).
/// </summary>
public static class HeadlessApp
{
    private static readonly Lazy<HeadlessUnitTestSession> s_session = new(() => HeadlessUnitTestSession.StartNew(typeof(HeadlessApp)));

    /// <summary>Used by <see cref="HeadlessUnitTestSession.StartNew(Type)"/> to configure the application.</summary>
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>()
        .UseSkia()
        .WithInterFont()
        .UseHeadless(new AvaloniaHeadlessPlatformOptions { UseHeadlessDrawing = false });

    public static Task RunAsync(Func<Task> action) => s_session.Value.Dispatch(async () =>
    {
        await action();
        return true;
    }, TestContext.Current.CancellationToken);

    /// <summary>Renders until <paramref name="condition"/> holds (at most about two seconds).</summary>
    public static async Task WaitForAsync(Window window, Func<bool> condition)
    {
        for (int i = 0; i < 40 && !condition(); i++)
        {
            await RenderAsync(window);
        }
    }

    /// <summary>Runs pending UI work and layout passes (the headless platform has no render loop driving layout).</summary>
    public static async Task RenderAsync(Window window)
    {
        for (int i = 0; i < 3; i++)
        {
            Dispatcher.UIThread.RunJobs();
            await Task.Delay(30);
            window.UpdateLayout();
        }

        Dispatcher.UIThread.RunJobs();
    }
}
