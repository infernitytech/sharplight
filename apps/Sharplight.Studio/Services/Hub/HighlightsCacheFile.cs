namespace Sharplight.Studio.Services.Hub;

/// <summary>The highlighted models as they are cached on disk between sessions.</summary>
/// <param name="FetchedAt">When the Hub was asked.</param>
/// <param name="Sections">The sections.</param>
internal sealed record HighlightsCacheFile(DateTimeOffset FetchedAt, IReadOnlyList<HighlightSection> Sections);
