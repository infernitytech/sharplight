namespace Sharplight.Distributed;

/// <summary>
/// A segment worker failed a request, closed its connection, stopped answering heartbeats or took longer than the timeout. The remote
/// segments of the connection are unusable afterwards: dispose what was compiled on them and load the model again (with the worker restarted
/// or replaced) to rebuild the session.
/// </summary>
public sealed class SegmentWorkerException : Exception
{
    /// <summary>Creates the exception.</summary>
    public SegmentWorkerException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public SegmentWorkerException(string message) : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the error that caused it.</summary>
    public SegmentWorkerException(string message, Exception innerException) : base(message, innerException)
    {
    }

    /// <summary>The worker, as <c>host:port</c>; empty when unknown.</summary>
    public string Worker { get; init; } = "";

    /// <summary>Whether the worker is still connected (it failed this request only, e.g. a compilation that ran out of memory).</summary>
    public bool WorkerAlive { get; init; }
}
