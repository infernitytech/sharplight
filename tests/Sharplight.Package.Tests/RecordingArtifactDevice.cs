using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Package.Tests;

/// <summary>
/// A device that runs on a CPU device but declares a code tier and a binary tier like a GPU and records one
/// <c>kernel/&lt;graph&gt;/&lt;segment&gt;</c> entry per compiled segment into the model-level cache, so tests can see which graphs a
/// precompilation compiled. Devices with the same driver share their binary tier (an identical GPU).
/// </summary>
/// <param name="inner">The CPU device that really compiles.</param>
/// <param name="driver">Driver version; part of the binary tier's descriptor.</param>
/// <param name="index">Device index (display name "Recording GPU index").</param>
public sealed class RecordingArtifactDevice(IComputeDevice inner, string driver = "1", int index = 0) : IComputeDevice
{
    public DeviceInfo Info { get; } = new()
    {
        Provider = "fake",
        Index = index,
        Name = $"Recording GPU {index}",
        Kind = DeviceKind.DiscreteGpu,
        Driver = driver,
        Features = new HashSet<string>(),
    };

    public long AvailableMemory => inner.AvailableMemory;

    public IReadOnlyList<ArtifactTarget> ArtifactTargets { get; } =
    [
        new ArtifactTarget(FakeArtifactDevice.Code(), [ArtifactDescriptorKeys.Target], ["kernel/"]),
        new ArtifactTarget(FakeArtifactDevice.Binary(driver), [ArtifactDescriptorKeys.Vendor, ArtifactDescriptorKeys.Device, ArtifactDescriptorKeys.Driver], ["fake/"]),
    ];

    public OpSupport Supports(Node node) => inner.Supports(node);

    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options)
    {
        // As the kernel compiler does: entries the cache already has (stored with the model) are served, not put again.
        var key = $"kernel/{segment.Graph.Name}/{segment.Index}";
        if (options.CacheFor?.Invoke(this) is { } cache && !cache.TryGet(key, out _))
        {
            cache.Put(key, new byte[] { 1 });
        }

        return inner.Compile(segment, options);
    }

    public DeviceProfile CreateProfile() => new()
    {
        Provider = Info.Provider,
        Name = Info.Name,
        CodeTarget = "spirv_1_6",
        SubgroupSize = 32,
        MaxSharedMemory = 49152,
    };

    public void Dispose()
    {
        // The inner device is shared by the tests.
    }
}
