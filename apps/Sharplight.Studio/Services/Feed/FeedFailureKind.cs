namespace Sharplight.Studio.Services.Feed;

/// <summary>Why the model feed could not be listed.</summary>
internal enum FeedFailureKind
{
    /// <summary>The feed could not be reached and no verified copy is cached.</summary>
    Offline,

    /// <summary>A signature, digest, expiry or rollback check failed: the feed is not trusted and nothing is listed from it.</summary>
    Verification,

    /// <summary>The feed's documents are malformed or an unsupported version.</summary>
    Format,
}
