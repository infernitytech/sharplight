using Sharplight.Distributed.Transport;

namespace Sharplight.Distributed;

/// <summary>
/// The remote segments of one compiled model. They fail together: once one worker fails (an error, a lost connection, a timeout) every
/// segment of the model throws the same <see cref="SegmentWorkerException"/>, and the other workers are told to abort what they are doing
/// (a worker waiting for a boundary value from a dead neighbour would otherwise wait for its timeout). The model is then disposed and loaded
/// again to rebuild the session.
/// </summary>
internal sealed class RemotePipeline
{
    private readonly CancellationTokenSource _failed = new();
    private readonly List<WorkerConnection> _connections = [];
    private readonly Lock _lock = new();
    private SegmentWorkerException? _failure;

    /// <summary>Cancelled when the pipeline fails.</summary>
    public CancellationToken Failed => _failed.Token;

    /// <summary>Why the pipeline failed, or null.</summary>
    public SegmentWorkerException? Failure => Volatile.Read(ref _failure);

    /// <summary>Adds a connection whose segments belong to the pipeline.</summary>
    public void Add(WorkerConnection connection)
    {
        lock (_lock)
        {
            if (!_connections.Contains(connection))
            {
                _connections.Add(connection);
            }
        }
    }

    /// <summary>Fails the pipeline (the first failure wins) and asks every other worker to abort its executions in progress.</summary>
    /// <returns>The pipeline's failure.</returns>
    public SegmentWorkerException Fail(SegmentWorkerException failure)
    {
        if (Interlocked.CompareExchange(ref _failure, failure, null) is { } first)
        {
            return first;
        }

        List<WorkerConnection> connections;
        lock (_lock)
        {
            connections = [.. _connections];
        }

        foreach (var connection in connections)
        {
            connection.Post(FrameKind.Abort, []);
        }

        _failed.Cancel();
        return failure;
    }

    /// <exception cref="SegmentWorkerException">The pipeline failed.</exception>
    public void ThrowIfFailed()
    {
        if (Failure is { } failure)
        {
            throw new SegmentWorkerException(failure.Message, failure) { Worker = failure.Worker };
        }
    }
}
