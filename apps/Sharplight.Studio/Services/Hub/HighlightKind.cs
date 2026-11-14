namespace Sharplight.Studio.Services.Hub;

/// <summary>A section of the models Discover shows before anything is searched.</summary>
internal enum HighlightKind
{
    /// <summary>The Hub's trending models.</summary>
    Trending,

    /// <summary>Recently published models that are already downloaded a lot.</summary>
    New,

    /// <summary>The most downloaded models of the last 30 days.</summary>
    Popular,
}
