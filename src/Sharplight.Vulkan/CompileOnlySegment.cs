using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Vulkan;

/// <summary>A segment compiled for a device profile (<see cref="VulkanProfileDevice"/>): its kernels exist, nothing else; it cannot run.</summary>
internal sealed class CompileOnlySegment(GraphSegment segment, int launches) : ICompiledSegment
{
    public GraphSegment Segment { get; } = segment;

    /// <summary>Kernel launches of the lowered program (diagnostics).</summary>
    public int Launches { get; } = launches;

    public long DeviceMemoryBytes => 0;

    public void SetInput(Value value, ReadOnlySpan<byte> data) => throw NotRunnable();

    public void Execute(RunParameters parameters) => throw NotRunnable();

    public void GetOutput(Value value, Span<byte> destination) => throw NotRunnable();

    /// <summary>Does nothing: the segment owns no state.</summary>
    public void ResetState()
    {
    }

    public void Dispose()
    {
    }

    private static NotSupportedException NotRunnable()
        => new("This segment was compiled for a device profile (offline precompilation) and cannot run.");
}
