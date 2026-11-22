namespace Sharplight.Distributed;

/// <summary>Counters of the executions of remote segments on one worker (diagnostics and benchmarks).</summary>
/// <param name="Executions">Segment executions whose reply was read.</param>
/// <param name="RoundTripMs">Time from sending each execution to reading its reply, summed (the coordinator's view).</param>
/// <param name="WorkerMs">Time the worker spent in those executions, summed (its inputs, the forward pass, its outputs).</param>
/// <param name="BytesSent">Bytes sent to the worker (frames with headers).</param>
/// <param name="BytesReceived">Bytes received from the worker.</param>
public readonly record struct RemoteSegmentStatistics(long Executions, double RoundTripMs, double WorkerMs, long BytesSent, long BytesReceived)
{
    /// <summary>Mean round trip minus worker time per execution: what the network and the protocol add per hop and step.</summary>
    public double OverheadMsPerExecution => Executions == 0 ? 0 : (RoundTripMs - WorkerMs) / Executions;
}
