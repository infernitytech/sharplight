using Sharplight.Graph;

namespace Sharplight.Runtime;

/// <summary>A compiled, device-resident part of a graph.</summary>
public interface ICompiledSegment : IDisposable
{
    /// <summary>The segment this object was compiled from.</summary>
    public GraphSegment Segment { get; }
    /// <summary>Uploads host data for a segment input (dense, logical layout).</summary>
    public void SetInput(Value value, ReadOnlySpan<byte> data);
    /// <summary>Records/executes the segment. Implementations may run asynchronously until an output is read.</summary>
    public void Execute(RunParameters parameters);
    /// <summary>
    /// Like <see cref="Execute(RunParameters)"/>, but the outputs in <paramref name="skipDownloads"/> are not read back afterwards, so a
    /// provider that stages outputs in host memory may skip copying them. The default ignores the hint.
    /// </summary>
    public void Execute(RunParameters parameters, IReadOnlySet<Value> skipDownloads) => Execute(parameters);
    /// <summary>
    /// Prepares the work of an <see cref="Execute(RunParameters, IReadOnlySet{Value})"/> with these symbol bindings and skipped outputs
    /// ahead of time without running it (a provider that records command buffers per shape records them now), so that the first step of
    /// that shape does not pay for it. Inputs and scalars are not read. The default does nothing.
    /// </summary>
    public void Prepare(RunParameters parameters, IReadOnlySet<Value> skipDownloads)
    {
    }
    /// <summary>Downloads a segment output (dense, logical layout) after <see cref="Execute(RunParameters)"/>.</summary>
    public void GetOutput(Value value, Span<byte> destination);
    /// <summary>Clears persistent state (KV caches, recurrent states) owned by this segment.</summary>
    public void ResetState();
    /// <summary>Whether <see cref="CopyStateRows"/> works.</summary>
    public bool CanCopyStateRows => false;
    /// <summary>
    /// Copies <paramref name="count"/> rows of the outermost dimension of each named state this segment holds, starting at row
    /// <paramref name="from"/>, to the rows starting at <paramref name="to"/> (dense logical layout, rows of fixed size; the ranges do not
    /// overlap), ordered after the work already submitted and before work submitted later. States the segment does not hold are ignored.
    /// </summary>
    /// <exception cref="NotSupportedException">The segment cannot copy state (<see cref="CanCopyStateRows"/> is false).</exception>
    public void CopyStateRows(IReadOnlyCollection<string> states, int from, int to, int count) => throw new NotSupportedException("This segment cannot copy state rows.");
    /// <summary>Device memory held by this segment (weights, state, activations).</summary>
    public long DeviceMemoryBytes { get; }
    /// <summary>
    /// Whether <see cref="Execute(RunParameters)"/> returns once the work is submitted, before the device finishes it (the segment waits
    /// before it reads an output or takes an input), so the host can do other work meanwhile. The default is false.
    /// </summary>
    public bool ExecutesAsynchronously => false;
    /// <summary>Where the most recent execution spent its time (diagnostics). The default reports zeros.</summary>
    public SegmentRunTimings LastRunTimings => default;
}
