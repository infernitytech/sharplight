namespace Sharplight.Runtime;

/// <summary>
/// A compiled segment that copies its weights in at every execution (<see cref="CompileOptions.StreamWeights"/>) and can start the copies
/// early: <see cref="CompiledModel"/> calls <see cref="BeginStreaming"/> before it runs the segment that precedes this one, so the copies
/// overlap that segment's work where the device can copy asynchronously.
/// </summary>
public interface IStreamedSegment : ICompiledSegment
{
    /// <summary>Starts copying the weights for the next execution; the execution waits for the copies (or makes them when none was started).</summary>
    public void BeginStreaming();

    /// <summary>
    /// Whether the streamed-weight memory still holds this segment's weights from its last copy, so that <see cref="BeginStreaming"/> and the
    /// next executions copy nothing and use them as they are (the memory is shared with the other streamed segments of the graph on the
    /// device, see <see cref="CompileOptions.StreamBuffer"/>). <see cref="CompiledModel"/> tracks which segment's weights each memory holds and
    /// sets it before it starts or runs the segment; false (the default) copies the weights in for every execution.
    /// </summary>
    public bool WeightsResident { get; set; }

    /// <summary>
    /// Whether <see cref="BeginStreaming"/> leaves the copies to the device (they overlap the host's work) rather than making them on the
    /// calling thread, for instance from page-locked sources. <see cref="CompiledModel"/> splits expert passes only onto segments that do.
    /// </summary>
    public bool CopiesAsynchronously => true;
}
