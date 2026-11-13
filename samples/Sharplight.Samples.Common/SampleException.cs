namespace Sharplight.Samples;

/// <summary>A user-facing failure (bad input, missing file, unavailable device). The message is printed without a stack trace.</summary>
public sealed class SampleException : Exception
{
    /// <summary>Creates the exception with the message shown to the user.</summary>
    public SampleException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with the message shown to the user and the underlying cause.</summary>
    public SampleException(string message, Exception inner)
        : base(message, inner)
    {
    }
}
