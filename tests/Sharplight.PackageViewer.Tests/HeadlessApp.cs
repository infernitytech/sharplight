using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Threading;
using Xunit;

namespace Sharplight.PackageViewer.Tests;

/// <summary>
/// The viewer's <see cref="App"/> on Avalonia's headless platform. Tests run their UI code with <see cref="RunAsync{T}"/> on the
/// session's UI thread (one session for the test run).
/// </summary>
public static class HeadlessApp
{
    private static readonly Lazy<HeadlessUnitTestSession> s_session = new(() => HeadlessUnitTestSession.StartNew(typeof(HeadlessApp)));

    /// <summary>Used by <see cref="HeadlessUnitTestSession.StartNew(Type)"/> to configure the application.</summary>
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>().UseHeadless(new AvaloniaHeadlessPlatformOptions());

    public static Task<T> RunAsync<T>(Func<Task<T>> action) => s_session.Value.Dispatch(action, TestContext.Current.CancellationToken);

    public static Task RunAsync(Func<Task> action) => RunAsync(async () =>
    {
        await action();
        return true;
    });

    /// <summary>Runs pending UI work and a layout pass (the headless platform has no render loop driving layout).</summary>
    public static void Render(Window window)
    {
        Dispatcher.UIThread.RunJobs();
        window.UpdateLayout();
        Dispatcher.UIThread.RunJobs();
    }
}
