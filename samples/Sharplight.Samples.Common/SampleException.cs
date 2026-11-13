namespace Sharplight.Samples;

/// <summary>A user-facing failure (bad input, missing file, unavailable device). The message is printed without a stack trace.</summary>
public sealed class SampleException : Exception
{
    /// <summary>Creates the exception with the message shown to the user and, optionally, the process exit code to report.</summary>
    public SampleException(string message, int exitCode = ExitCodes.RuntimeError)
        : base(message)
    {
        ExitCode = exitCode;
    }

    /// <summary>Creates the exception with the message shown to the user and the underlying cause.</summary>
    public SampleException(string message, Exception inner)
        : base(message, inner)
    {
        ExitCode = ExitCodes.RuntimeError;
    }

    /// <summary>The process exit code to report; one of <see cref="ExitCodes"/>.</summary>
    public int ExitCode { get; }
}
