namespace Sharplight.ModelFeed;

/// <summary>A feed document or package is malformed (bad JSON, invalid id or version, missing field).</summary>
public class FeedFormatException : Exception
{
    /// <summary>Creates the exception.</summary>
    public FeedFormatException()
    {
    }

    /// <summary>Creates the exception.</summary>
    /// <param name="message">What is wrong.</param>
    public FeedFormatException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception.</summary>
    /// <param name="message">What is wrong.</param>
    /// <param name="innerException">The cause.</param>
    public FeedFormatException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
