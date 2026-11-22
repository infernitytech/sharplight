using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Cpu.Tests;

/// <summary>A device that runs as <paramref name="inner"/> and reports <paramref name="overhead"/> bytes per compiled segment (placement tests).</summary>
internal sealed class SegmentOverheadDevice(IComputeDevice inner, long overhead) : IComputeDevice
{
    /// <inheritdoc/>
    public DeviceInfo Info => inner.Info;

    /// <inheritdoc/>
    public long AvailableMemory => inner.AvailableMemory;

    /// <inheritdoc/>
    public long SegmentOverheadBytes => overhead;

    /// <inheritdoc/>
    public OpSupport Supports(Node node) => inner.Supports(node);

    /// <inheritdoc/>
    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options) => inner.Compile(segment, options);

    /// <inheritdoc/>
    public void Dispose()
    {
    }
}
