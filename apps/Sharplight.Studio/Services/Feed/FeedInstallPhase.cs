namespace Sharplight.Studio.Services.Feed;

/// <summary>The step of a feed install.</summary>
internal enum FeedInstallPhase
{
    /// <summary>Reading the feed and choosing the versions of the package and its dependencies.</summary>
    Resolving,

    /// <summary>Downloading and verifying a package file.</summary>
    Downloading,

    /// <summary>Replacing the stored package and recording it.</summary>
    Installing,

    /// <summary>Everything is installed.</summary>
    Done,
}
