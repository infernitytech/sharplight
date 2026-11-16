namespace Sharplight.Storage;

/// <summary>Where Sharplight keeps the files it writes (see <see cref="StorageLayout"/>).</summary>
public enum StorageMode
{
    /// <summary>The platform's per-user folders (XDG on Linux, <c>%APPDATA%</c>/<c>%LOCALAPPDATA%</c> on Windows, <c>~/Library</c> on macOS).</summary>
    System,

    /// <summary>
    /// A <c>data</c> folder next to the package (the AppImage, the Velopack install root, the <c>.app</c> bundle or the executable's
    /// folder), so the application and everything it wrote can be moved together.
    /// </summary>
    Portable,

    /// <summary>One folder chosen by the user.</summary>
    Custom,
}
