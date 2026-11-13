namespace Sharplight.Runtime;

/// <summary>
/// Well-known keys of artifact descriptors (<see cref="ArtifactTarget.Descriptor"/>). Every descriptor has
/// <see cref="Provider"/>, <see cref="Tier"/> and <see cref="Kind"/>; the other keys are provider defined.
/// </summary>
public static class ArtifactDescriptorKeys
{
    /// <summary>Name of the execution provider the artifacts are for (<see cref="DeviceInfo.Provider"/>, e.g. <c>vulkan</c>).</summary>
    public const string Provider = "provider";

    /// <summary>Artifact tier, see <see cref="ArtifactTiers"/>.</summary>
    public const string Tier = "tier";

    /// <summary>What the entries are (e.g. <c>spirv</c>, <c>pipeline-cache</c>).</summary>
    public const string Kind = "kind";

    /// <summary>Code generation target of a code tier (e.g. <c>spirv1.6</c>); for legacy kernel sets the version 1 device target string.</summary>
    public const string Target = "target";

    /// <summary>Lanes per subgroup the code was specialized for.</summary>
    public const string Subgroup = "subgroup";

    /// <summary>Shared (workgroup) memory per workgroup in bytes the code was specialized for.</summary>
    public const string SharedMemory = "shared";

    /// <summary>Comma-separated, ordinally sorted device features (<see cref="DeviceFeatures"/>) the code may use.</summary>
    public const string Features = "features";

    /// <summary>Compiler that produced code-tier entries (name and version).</summary>
    public const string Compiler = "compiler";

    /// <summary>Hardware vendor id (hexadecimal) of a binary tier.</summary>
    public const string Vendor = "vendor";

    /// <summary>Device model of a binary tier (hexadecimal PCI device id); every card of that model shares it.</summary>
    public const string Device = "device";
    /// <summary>
    /// Driver cache compatibility id of a binary tier (Vulkan pipelineCacheUUID): changes with the driver build and decides
    /// whether the driver accepts the stored binaries.
    /// </summary>
    public const string CacheId = "cache";

    /// <summary>Driver version (hexadecimal) of a binary tier.</summary>
    public const string Driver = "driver";

    /// <summary>Kind of legacy (version 1) kernel sets, whose descriptor is synthesized when they are read.</summary>
    public const string LegacyKernelSetKind = "kernel-set";
}
