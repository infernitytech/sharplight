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
    /// <summary>Downloads a segment output (dense, logical layout) after <see cref="Execute(RunParameters)"/>.</summary>
    public void GetOutput(Value value, Span<byte> destination);
    /// <summary>Clears persistent state (KV caches, recurrent states) owned by this segment.</summary>
    public void ResetState();
    /// <summary>Whether <see cref="CopyStateRows"/> works.</summary>
    public bool CanCopyStateRows => false;
    /// <summary>
    /// Copies row <paramref name="from"/> of the outermost dimension of each named state this segment holds to row <paramref name="to"/>
    /// (dense logical layout, rows of fixed size), ordered after the work already submitted and before work submitted later. States
    /// the segment does not hold are ignored.
    /// </summary>
    /// <exception cref="NotSupportedException">The segment cannot copy state (<see cref="CanCopyStateRows"/> is false).</exception>
    public void CopyStateRows(IReadOnlyCollection<string> states, int from, int to) => throw new NotSupportedException("This segment cannot copy state rows.");
    /// <summary>Device memory held by this segment (weights, state, activations).</summary>
    public long DeviceMemoryBytes { get; }
}
