using Sharplight.Runtime;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>Readable labels and values for artifact descriptor keys (<see cref="ArtifactDescriptorKeys"/>).</summary>
internal static class ArtifactDescriptorLabels
{
    /// <summary>The label of a key; the key itself for keys this version does not know.</summary>
    public static string Label(string key) => key switch
    {
        ArtifactDescriptorKeys.Provider => "Provider",
        ArtifactDescriptorKeys.Tier => "Tier",
        ArtifactDescriptorKeys.Kind => "Kind",
        ArtifactDescriptorKeys.Target => "Code target",
        ArtifactDescriptorKeys.Subgroup => "Subgroup size",
        ArtifactDescriptorKeys.SharedMemory => "Shared memory",
        ArtifactDescriptorKeys.Features => "Device features",
        ArtifactDescriptorKeys.Compiler => "Compiler",
        ArtifactDescriptorKeys.Vendor => "Vendor id",
        ArtifactDescriptorKeys.Device => "Device id",
        ArtifactDescriptorKeys.CacheId => "Cache id",
        ArtifactDescriptorKeys.Driver => "Driver version",
        ArtifactDescriptorKeys.DeviceName => "Device",
        ArtifactDescriptorKeys.KernelAbi => "Kernel ABI",
        ArtifactDescriptorKeys.BinaryKey => "Binary key",
        _ => key,
    };

    /// <summary>The value for display: shared memory as a size, feature lists with spaces, the rest as stored.</summary>
    public static string Value(string key, string value) => key switch
    {
        ArtifactDescriptorKeys.SharedMemory when long.TryParse(value, out long bytes) => ByteFormat.Format(bytes),
        ArtifactDescriptorKeys.Features => value.Length == 0 ? "none" : value.Replace(",", ", ", StringComparison.Ordinal),
        _ => value,
    };
}
