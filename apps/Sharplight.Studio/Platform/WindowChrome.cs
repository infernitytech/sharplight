namespace Sharplight.Studio.Platform;

/// <summary>
/// Chooses between Studio's own title bar and the system's. The own title bar moves, resizes, maximizes and snaps the window by asking
/// the window manager (<c>_NET_WM_MOVERESIZE</c> on X11, which KWin, Mutter and most others implement, also for XWayland windows), so it
/// is used only where that request is known to work; everywhere else, and whenever the user asks for it, the system frames the window.
/// </summary>
internal static class WindowChrome
{
    /// <summary>
    /// Overrides the choice: <c>system</c> for the system title bar, <c>custom</c> for Studio's own one even where the window manager
    /// was not detected.
    /// </summary>
    public const string EnvironmentVariable = "SHARPLIGHT_STUDIO_TITLE_BAR";

    /// <summary>The chrome for this session.</summary>
    /// <param name="useSystemTitleBar">The setting that asks for the system title bar.</param>
    /// <param name="environment">The value of <see cref="EnvironmentVariable"/>, or null.</param>
    /// <param name="isLinux">The app runs on Linux (the only platform whose own title bar is implemented and tested).</param>
    /// <param name="windowManagerMovesWindows">Whether the window manager moves and resizes windows on request (asked only when needed).</param>
    public static WindowChromeMode Choose(bool useSystemTitleBar, string? environment, bool isLinux, Func<bool> windowManagerMovesWindows)
    {
        switch (environment?.Trim().ToLowerInvariant())
        {
            case "system":
                return WindowChromeMode.System;
            case "custom":
                return WindowChromeMode.Custom;
        }

        if (useSystemTitleBar || !isLinux)
        {
            return WindowChromeMode.System;
        }

        return windowManagerMovesWindows() ? WindowChromeMode.Custom : WindowChromeMode.System;
    }

    /// <summary>The chrome for this process, from the settings, the environment and the running window manager.</summary>
    /// <param name="useSystemTitleBar">The setting that asks for the system title bar.</param>
    public static WindowChromeMode ForThisProcess(bool useSystemTitleBar)
        => Choose(useSystemTitleBar, Environment.GetEnvironmentVariable(EnvironmentVariable), OperatingSystem.IsLinux(), X11WindowManager.SupportsMoveResize);
}
