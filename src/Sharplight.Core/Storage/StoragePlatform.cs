namespace Sharplight.Storage;

/// <summary>The platform conventions a <see cref="StorageLayout"/> follows in <see cref="StorageMode.System"/>.</summary>
public enum StoragePlatform
{
    /// <summary>Linux and other Unix systems: the XDG base directories.</summary>
    Linux,

    /// <summary>Windows: <c>%APPDATA%</c> and <c>%LOCALAPPDATA%</c>.</summary>
    Windows,

    /// <summary>macOS: <c>~/Library/Application Support</c>, <c>~/Library/Caches</c> and <c>~/Library/Logs</c>.</summary>
    MacOS,
}
