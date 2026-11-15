using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Package.Tests;

/// <summary>
/// A device that runs on a CPU device but declares a code tier like a GPU and records one <c>kernel/&lt;graph&gt;/&lt;segment&gt;</c>
/// entry per compiled segment into the model-level cache, so tests can see which graphs a precompilation compiled.
/// </summary>
public sealed class RecordingArtifactDevice(IComputeDevice inner) : IComputeDevice
{
    public DeviceInfo Info { get; } = new() { Provider = "fake", Index = 0, Name = "Recording GPU", Kind = DeviceKind.DiscreteGpu, Driver = "1" };

    public long AvailableMemory => inner.AvailableMemory;

    public IReadOnlyList<ArtifactTarget> ArtifactTargets { get; } = [new ArtifactTarget(FakeArtifactDevice.Code(), [ArtifactDescriptorKeys.Target], ["kernel/"])];

    public OpSupport Supports(Node node) => inner.Supports(node);

    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options)
    {
        options.CacheFor?.Invoke(this)?.Put($"kernel/{segment.Graph.Name}/{segment.Index}", new byte[] { 1 });
        return inner.Compile(segment, options);
    }

    public void Dispose()
    {
        // The inner device is shared by the tests.
    }
}
