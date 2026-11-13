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
    /// <summary>Device memory held by this segment (weights, state, activations).</summary>
    public long DeviceMemoryBytes { get; }
}
