namespace Sharplight.Studio.Services.Feed;

/// <summary>A feed that could not be listed.</summary>
/// <param name="Kind">The category.</param>
/// <param name="Message">What went wrong, for the user.</param>
internal sealed record FeedFailure(FeedFailureKind Kind, string Message);
