namespace Sharplight.Studio.Services.Feed;

/// <summary>A feed install that cannot proceed for a reason other than a failed verification (no compatible version, a missing dependency).</summary>
internal sealed class FeedInstallException : Exception
{
    public FeedInstallException()
    {
    }

    public FeedInstallException(string message)
        : base(message)
    {
    }

    public FeedInstallException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
