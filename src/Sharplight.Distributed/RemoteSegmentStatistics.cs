namespace Sharplight.Distributed;

/// <summary>Counters of the executions of remote segments on one worker (diagnostics and benchmarks).</summary>
/// <param name="Executions">Segment executions whose reply was read.</param>
/// <param name="RoundTripMs">Time from sending each execution to reading its reply, summed (the coordinator's view).</param>
/// <param name="WorkerMs">Time the worker spent in those executions, summed (the forward pass and reading its outputs).</param>
/// <param name="WaitMs">
/// Time the executions waited on the worker before they ran, summed: behind earlier work of the worker, and for the boundary values of the
/// neighbour before (its compute and the hop between the two workers).
/// </param>
/// <param name="BytesSent">Bytes sent to the worker (frames with headers).</param>
/// <param name="BytesReceived">Bytes received from the worker.</param>
public readonly record struct RemoteSegmentStatistics(long Executions, double RoundTripMs, double WorkerMs, double WaitMs, long BytesSent, long BytesReceived)
{
    /// <summary>
    /// Mean round trip minus the worker's compute and waits per execution: what the network and the protocol add to one execution's
    /// request and reply (two traversals between the coordinator and the worker).
    /// </summary>
    public double OverheadMsPerExecution => Executions == 0 ? 0 : (RoundTripMs - WorkerMs - WaitMs) / Executions;
}
