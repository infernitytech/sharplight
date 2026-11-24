namespace Sharplight.Runtime;

/// <summary>
/// A compiled segment that copies its weights in at every execution (<see cref="CompileOptions.StreamWeights"/>) and can start the copies
/// early: <see cref="CompiledModel"/> calls <see cref="BeginStreaming()"/> before it runs the segment that precedes this one, so the copies
/// overlap that segment's work where the device can copy asynchronously.
/// </summary>
public interface IStreamedSegment : ICompiledSegment
{
    /// <summary>Starts copying the weights for the next execution; the execution waits for the copies (or makes them when none was started).</summary>
    public void BeginStreaming();

    /// <summary>
    /// Whether <see cref="BeginStreaming(ReadOnlySpan{int})"/> copies only the experts a pass selects of the segment's expert banks (the
    /// <c>MatMulId</c> weights indexed by its one ids input). False (the default): it copies every weight, as <see cref="BeginStreaming()"/>.
    /// </summary>
    public bool StreamsSelectedExperts => false;

    /// <summary>
    /// Starts copying the weights for the next execution, of the expert banks only the experts in <paramref name="expertIds"/> (the pass's
    /// ids input, i32 [rows, slots]; ids outside a bank select nothing) at their places in the bank; the other experts of the memory keep
    /// what it held, so the weights are not resident afterwards (<see cref="WeightsResident"/>). The pass must read no other experts.
    /// </summary>
    /// <param name="expertIds">The expert ids of the pass the execution runs.</param>
    public void BeginStreaming(ReadOnlySpan<int> expertIds) => BeginStreaming();

    /// <summary>
    /// Whether the streamed-weight memory still holds this segment's weights from its last copy, so that <see cref="BeginStreaming()"/> and the
    /// next executions copy nothing and use them as they are (the memory is shared with the other streamed segments of the graph on the
    /// device, see <see cref="CompileOptions.StreamBuffer"/>). <see cref="CompiledModel"/> tracks which segment's weights each memory holds and
    /// sets it before it starts or runs the segment; false (the default) copies the weights in for every execution.
    /// </summary>
    public bool WeightsResident { get; set; }

    /// <summary>
    /// Whether <see cref="BeginStreaming()"/> leaves the copies to the device (they overlap the host's work) rather than making them on the
    /// calling thread, for instance from page-locked sources. <see cref="CompiledModel"/> splits expert passes only onto segments that do.
    /// </summary>
    public bool CopiesAsynchronously => true;
}
