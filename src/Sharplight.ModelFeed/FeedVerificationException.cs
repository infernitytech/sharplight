namespace Sharplight.ModelFeed;

/// <summary>
/// A feed object failed verification: bad or missing signature, untrusted or revoked key, digest or hash mismatch, an expired index
/// (freeze) or an older sequence than already seen (rollback). Never retried against the same source.
/// </summary>
public class FeedVerificationException : Exception
{
    /// <summary>Creates the exception.</summary>
    public FeedVerificationException()
    {
    }

    /// <summary>Creates the exception.</summary>
    /// <param name="message">What failed.</param>
    public FeedVerificationException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception.</summary>
    /// <param name="message">What failed.</param>
    /// <param name="innerException">The cause.</param>
    public FeedVerificationException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
