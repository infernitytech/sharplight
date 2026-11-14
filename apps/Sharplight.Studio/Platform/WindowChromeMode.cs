namespace Sharplight.Studio.Platform;

/// <summary>How the main window is framed.</summary>
internal enum WindowChromeMode
{
    /// <summary>Studio's own title bar (icon, title, minimize / maximize / close) and resize edges; the window manager draws nothing.</summary>
    Custom,

    /// <summary>The window manager's title bar and frame.</summary>
    System,
}
