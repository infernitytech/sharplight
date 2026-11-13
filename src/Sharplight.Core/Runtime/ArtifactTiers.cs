namespace Sharplight.Runtime;

/// <summary>
/// Well-known tiers of precompiled artifacts (the <see cref="ArtifactDescriptorKeys.Tier"/> of an artifact set). Tiers are plain
/// strings: a provider may define more, and readers skip sets of tiers no device asks for.
/// </summary>
public static class ArtifactTiers
{
    /// <summary>
    /// Intermediate device code that does not depend on the driver (SPIR-V for Vulkan, later PTX, DXIL, ...). Keyed by a device
    /// profile (provider, code target, limits, features) rather than by a physical device.
    /// </summary>
    public const string Code = "code";

    /// <summary>Driver-specific binaries (the Vulkan driver pipeline cache, later CUBIN, ...). Keyed by device identity and driver version.</summary>
    public const string Binary = "binary";

    /// <summary>
    /// Kernel sets of version 1 packages, which mixed code and binaries for one device target
    /// (<see cref="IComputeDevice.CompilationTarget"/>). Read only; consulted for every tier of a device with that target.
    /// </summary>
    public const string Mixed = "mixed";
}
