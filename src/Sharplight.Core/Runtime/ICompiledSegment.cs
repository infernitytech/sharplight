using Sharplight.Graph;

namespace Sharplight.Runtime;

/// <summary>A compiled, device-resident part of a graph.</summary>
public interface ICompiledSegment : IDisposable
{
    public GraphSegment Segment { get; }
    /// <summary>Uploads host data for a segment input (dense, logical layout).</summary>
    public void SetInput(Value value, ReadOnlySpan<byte> data);
    /// <summary>Records/executes the segment. Implementations may run asynchronously until an output is read.</summary>
    public void Execute(RunParameters parameters);
    /// <summary>Downloads a segment output (dense, logical layout) after <see cref="Execute"/>.</summary>
    public void GetOutput(Value value, Span<byte> destination);
    /// <summary>Clears persistent state (KV caches, recurrent states) owned by this segment.</summary>
    public void ResetState();
    /// <summary>Device memory held by this segment (weights, state, activations).</summary>
    public long DeviceMemoryBytes { get; }
}
