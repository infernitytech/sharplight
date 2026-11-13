using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Package.Tests;

/// <summary>
/// A device that only describes artifact targets the way the Vulkan device does: a code tier (<c>kernel/*</c>, matched on the code
/// target) and a binary tier (everything else, matched on vendor, device and driver). Cannot compile.
/// </summary>
public sealed class FakeArtifactDevice(string codeTarget = "spirv_1_6", string driver = "1", int subgroup = 32) : IComputeDevice
{
    public const string BinaryKey = "fake/pipeline_cache";

    public DeviceInfo Info { get; } = new() { Provider = "fake", Index = 0, Name = "Fake GPU", Kind = DeviceKind.DiscreteGpu, Driver = driver };

    public long AvailableMemory => 0;

    public IReadOnlyList<ArtifactTarget> ArtifactTargets { get; } =
    [
        new ArtifactTarget(Code(codeTarget, subgroup), [ArtifactDescriptorKeys.Target], ["kernel/"]),
        new ArtifactTarget(Binary(driver), [ArtifactDescriptorKeys.Vendor, ArtifactDescriptorKeys.Device, ArtifactDescriptorKeys.Driver], []),
    ];

    public int ExportCalls { get; private set; }

    public static Dictionary<string, string> Code(string target = "spirv_1_6", int subgroup = 32, string provider = "fake") => new()
    {
        [ArtifactDescriptorKeys.Provider] = provider,
        [ArtifactDescriptorKeys.Tier] = ArtifactTiers.Code,
        [ArtifactDescriptorKeys.Kind] = "spirv",
        [ArtifactDescriptorKeys.Target] = target,
        [ArtifactDescriptorKeys.Subgroup] = subgroup.ToString(System.Globalization.CultureInfo.InvariantCulture),
    };

    public static Dictionary<string, string> Binary(string driver = "1", string provider = "fake") => new()
    {
        [ArtifactDescriptorKeys.Provider] = provider,
        [ArtifactDescriptorKeys.Tier] = ArtifactTiers.Binary,
        [ArtifactDescriptorKeys.Kind] = "pipeline-cache",
        [ArtifactDescriptorKeys.Vendor] = "10de",
        [ArtifactDescriptorKeys.Device] = "uuid-1",
        [ArtifactDescriptorKeys.Driver] = driver,
    };

    public OpSupport Supports(Node node) => throw new NotSupportedException();

    public ICompiledSegment Compile(GraphSegment segment, CompileOptions options) => throw new NotSupportedException();

    public void ExportCompilationState(ICompilationCache cache)
    {
        ExportCalls++;
        cache.Put(BinaryKey, new byte[] { 42 });
    }

    public void Dispose()
    {
    }
}
