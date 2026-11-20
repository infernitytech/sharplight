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
}
