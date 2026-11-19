namespace Sharplight.Studio.ViewModels.Models;

/// <summary>How the library orders its models.</summary>
internal enum LibrarySort
{
    /// <summary>By name.</summary>
    Name,

    /// <summary>Largest first.</summary>
    Size,

    /// <summary>Most recently used first.</summary>
    LastUsed,

    /// <summary>Not used for the longest time first (the cleanup candidates).</summary>
    LeastRecentlyUsed,
}
