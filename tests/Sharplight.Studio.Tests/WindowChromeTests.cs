using Avalonia.Controls;
using Sharplight.Studio.Platform;
using Sharplight.Studio.Views;

namespace Sharplight.Studio.Tests;

/// <summary>The choice between Studio's own title bar and the system's, and the window's frame for each.</summary>
public sealed class WindowChromeTests
{
    [Fact]
    public void Choose_UsesTheOwnTitleBarOnlyWhereTheWindowManagerMovesWindows()
    {
        int asked = 0;
        bool Moves()
        {
            asked++;
            return true;
        }

        Assert.Equal(WindowChromeMode.Custom, WindowChrome.Choose(false, null, isLinux: true, Moves));
        Assert.Equal(WindowChromeMode.System, WindowChrome.Choose(false, null, isLinux: true, () => false));
        // Not implemented elsewhere, and the setting wins; neither asks the window manager.
        asked = 0;
        Assert.Equal(WindowChromeMode.System, WindowChrome.Choose(false, null, isLinux: false, Moves));
        Assert.Equal(WindowChromeMode.System, WindowChrome.Choose(true, null, isLinux: true, Moves));
        Assert.Equal(0, asked);
        // The environment overrides both ways.
        Assert.Equal(WindowChromeMode.System, WindowChrome.Choose(false, "system", isLinux: true, Moves));
        Assert.Equal(WindowChromeMode.Custom, WindowChrome.Choose(true, " Custom ", isLinux: false, () => false));
        Assert.Equal(WindowChromeMode.Custom, WindowChrome.Choose(false, "other", isLinux: true, Moves));
    }

    [Fact]
    public Task Window_ShowsItsOwnTitleBarAndEdges_OnlyWithCustomChrome() => HeadlessApp.RunAsync(async () =>
    {
        var window = new MainWindow { Width = 1200, Height = 800 };
        window.Show();
        try
        {
            await HeadlessApp.RenderAsync(window);
            var titleBar = window.FindControl<Border>("TitleBar")!;
            var edges = window.FindControl<Panel>("ResizeEdges")!;
            Assert.Equal(WindowChromeMode.Custom, window.Chrome);
            Assert.Equal(WindowDecorations.None, window.WindowDecorations);
            Assert.True(titleBar.IsVisible);
            Assert.True(edges.IsVisible);

            // Maximized: no edges; the maximize button restores. A double-click toggles the same way.
            window.WindowState = WindowState.Maximized;
            Assert.False(edges.IsVisible);
            Assert.Equal("Restore", ToolTip.GetTip(window.FindControl<Button>("MaximizeButton")!));
            window.WindowState = WindowState.Normal;
            Assert.True(edges.IsVisible);

            window.ApplyChrome(WindowChromeMode.System);
            Assert.Equal(WindowDecorations.Full, window.WindowDecorations);
            Assert.False(titleBar.IsVisible);
            Assert.False(edges.IsVisible);
        }
        finally
        {
            window.Close();
        }
    });
}
